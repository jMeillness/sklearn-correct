class ConfidenceRankError(Exception):
    """ Raise if errors occurs in confidence ranking.
    """
    pass


class ConfidenceUnsetError(Exception):
    """ Raise when an unset confidence value is used.
    """
    pass


class Feature(object):
    """ A candidate suggestion feature that scores correction candidate.
        
    Attributes:
        name: the string representation of the feature. It is identical to the
            return value of 'self.__str__()'.
    """

    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name


class FeatureRegistry(object):
    """ An object record the suggestion features.

    This object assigns a unique integer ID to a feature, starting from 0.
    It supports ID lookup by name and vise versa.

    Attributes:
        __list: a list of features.
        __dict: a dictionary stores the mappings from feature to the according
            ID.
    """

    def __init__(self):
        self.__list = []
        self.__dict = {}

    def register(self, feature):
        """ Add feature into this registry.
        
        Args:
            feature: a feature object.
        
        Raises:
            ValueError: If the given feature have the identical name with a
                ready exist feature of this registry.
        """
        if feature.name in self.__dict:
            raise ValueError('\'%s\' already exists in the registry: %s' %
                    (feature.name, [str(f) for f in self.__list]))
        self.__list.append(feature)
        self.__dict[feature.name] = len(self.__list) - 1

    def get(self, key):
        """ Get the according feature information.
        
        The parameter can be the either the feature (name) or the assigned
        feature ID, this method will returns the other one.
        
        Args:
            key: An object indicates the feature, a string indicates the feature
                name, or an integer indicates the feature ID.
        
        Returns:
            The according feature object if 'key' is the feature ID. Or the
            feature ID if 'key' is the feature object or feature name.
        
        Raises:
            KeyError: If the given key do not exist.
            TypeError: If the type of 'key' is neither 'Feature', 'int' or
                'str'.
        """
        if isinstance(key, str):
            return self.__dict[key]
        elif isinstance(key, Feature):
            return self.__dict[key.name]
        elif isinstance(key, int):
            try:
                return self.__list[key]
            except IndexError as e:
                raise KeyError(e)
        else:
            raise TypeError

    def size(self):
        return len(self.__list)


class Candidate(object):
    """ An abstract error candidate.

    Attributes:
        name: a string indicates the candidate text.
        label: a boolean value indicates whether the candidate is a correction
            of the error.
        feature_values: a list of float values. The index of the value in the
            list is according to the feature ID in the registry.
        confidence: a float in range 0-1, which indicates the probability of
            this candidate to be a correction. If the confidence is unset, it
            has the default value -1.
    """

    def __init__(self, name, feature_values, label):
        self.name = name
        self.feature_values = feature_values
        self.label = label
        self.confidence = -1


class WeightingMixin:

    @property
    def feature_weights(self):
        num_true = sum([1 if l else 0 for l in self.labels])
        true_weight = float(len(self.labels) - num_true) / num_true
        return [true_weight if l else 1 for l in self.labels]


class Error(WeightingMixin, object):
    """ An abstract error.

    An abstract error contains a list of suggested candidates.

    Attributes:
        name: a string indicates the error text.
        candidates: a list of candidates objects suggested for correcting this
            error.
        feature_value: a two-dimensional list of float, which indicates the
            feature values from all error candidates.
        labels: a list of integers, which indicates the labels of each
            error candidate.
    """

    def __init__(self, name, candidates=None):
        self.name = name
        self.candidates = [] if candidates == None else candidates

    @property
    def feature_values(self):
        """ Get the feature values of all candidates.
        
        Returns:
            A two dimensional list of floats. The feature value of each candidate
            stores in a nested list.
        """
        return [c.feature_values for c in self.candidates]

    @property
    def labels(self):
        """ Get the labels of all candidates.
        
        Returns:
            A list of integers, which element is the feature value of a candidate.
        """
        return [c.label for c in self.candidates]

    @property
    def confidences(self):
        """ Get the confidences of all candidates.
        
        Returns:
            A list of floats, which element is the confidence of a candidate.
        """
        return [c.confidence for c in self.candidates]

    @confidences.setter
    def confidences(self, values):
        """ Set the confidences of all candidates.
        
            Values: a list of floats, which element is the confidence of a
                candidate.
        """
        if len(values) != len(self.candidates):
            raise ValueError('%d confidence values to be set to %d candidates' %
                    (len(values), len(self.candidates)))
        for i, c in enumerate(self.candidates):
            c.confidence = values[i]

    @property
    def rank(self):
        """ Get the rank of the top ranked correction among candidates sorted by
        confidences in a descending order. The rank starts from 1.
        
        Returns:
            A integer indicates the rank of the top ranked correction.

        Raises:
            ConfidenceRankError: if there is no candidate exists in this error.
            ConfidenceUnsetError: if an unset confidence value exists in any of
                the candidates.
        """
        if len(self.confidences) == 0:
            raise ConfidenceRankError
        if min(self.confidences) == -1:
            raise ConfidenceUnsetError
        conf = max([c.confidence if c.label else 0 for c in self.candidates])
        pos = sum([1 if c.confidence > conf else 0 for c in self.candidates])
        return pos + 1

    def add(self, candidate):
        """ Add candidate to this error.
        
        Args:
            candidate: a candidate object.
        """
        self.candidates.append(candidate)


class Dataset(WeightingMixin, object):
    """ An abstract suggestion dataset.

    An abstract dataset contains a list of errors.

    Attributes:
        feature_registry: a feature registry.
        errors: a list of errors.
    """

    def __init__(self, errors, feature_registry):
        self.feature_registry = feature_registry
        self.errors = errors

    @property
    def feature_values(self):
        """ Get the feature values of all candidates from all errors.
        
        Returns:
            A two dimensional list of floats. The feature value of each candidate
            stores in a nested list.
        """
        return reduce(lambda x, y: x + y,
                [e.feature_values for e in self.errors])

    @property
    def labels(self):
        """ Get the labels of all candidates from all errors.
        
        Returns:
            A list of integers, which element is the feature value of a candidate.
        """
        return reduce(lambda x, y: x + y, [e.labels for e in self.errors])

    @property
    def confidences(self):
        """ Get the confidences of all candidates of all errors.
        
        Returns:
            A list of floats, which element is the confidence of a candidate.
        """
        return reduce(lambda x, y: x + y, [e.confidences for e in self.errors])

    @confidences.setter
    def confidences(self, values):
        """ Set the confidences of all candidates of all errors.
        
            Values: a list of floats, which element is the confidence of a
                candidate.
        """
        used = 0
        for e in self.errors:
            num_candidates = len(e.candidates)
            e.confidences = values[used: used + num_candidates]
            used += num_candidates

    def subset(self, **kwargs):
        """ Generate a data subset.
        
        A data subset has the same feature set and a subset of errors.
        
        kwargs:
            filt: a filter function.
            first: a integer or float value indicates the number or the
                percentage of errors from the head of the error list is used to
                construct subset.
            last: same as first, but pick errors from the tail of the error
                list.
            return_complement: a boolean indicates whether the complement subset
                should also return.
        
        Returns:
            A dataset contains the same feature set and a subset of errors.
        """
        filt = kwargs.get('filt', None)
        first = kwargs.get('first', None)
        last = kwargs.get('last', None)
        return_complement = kwargs.get('return_complement', False)

        if sum([0 if arg is None else 1 for arg in [filt, first, last]]) != 1:
            raise ValueError

        if filt is not None:
            sub = Dataset(filter(filt, self.errors), self.feature_registry)
        elif first is not None:
            size = int(first * len(self.errors)) if first < 1 else first
            sub = Dataset(self.errors[: size], self.feature_registry)
        elif last is not None:
            size = int(last * len(self.errors)) if last < 1 else last
            sub = Dataset(
                    self.errors[len(self.errors) - size: ],
                    self.feature_registry)

        if return_complement:
            complement = Dataset(
                    filter(lambda err: err not in sub.errors, self.errors),
                    self.feature_registry)
            return sub, complement
        else:
            return sub

    def first(error_percentage):
        """ Generate a data subset. """

    @staticmethod
    def read(pathname):
        """ Read and construct a dataset using data from file.
        
        A valid data file should formated follows:
        -   the file starts with a list of feature names, which are separated by
            a newline character. Features follow by an empty line.
        -   each error starts with one line containing its name, following
            by its candidates.
        -   each error candidate uses one line, containing the following fields:
            candidate name, a list of candidate values, and a label. Fields are
            seperated by a tab characters.
        -   there is an empty line between each errors and 
        
        Args:
            pathname: a string indicates the pathname to a file.
        
        Returns:
            a dataset object with data from file.
        """
        feature_registry = FeatureRegistry()
        errors = []
        
        with open(pathname, 'r') as file:
            is_feature_line = True
            curr_error = None
         
            for line in file:
                line = line.strip()
                
                if is_feature_line:
                    # Read the feature line.
                    if len(line) == 0:
                        is_feature_line = False
                        feature_size = feature_registry.size()
                    else:
                        feature_registry.register(Feature(line))
                    
                else:
                    # Read data lines.
                    if len(line) == 0:
                        errors.append(curr_error)
                    else:
                        fields = line.split('\t')
                        if len(fields) == 1:
                            curr_error = Error(fields[0])
                        else:
                            curr_error.add(Candidate(
                                fields[0],
                                [float(f) for f in fields[1:-1]],
                                int(fields[-1])
                                ))
            
            if curr_error != None:
                errors.append(curr_error)
        
        return Dataset(errors, feature_registry)

    def precision_at(self, n=float('inf')):
        """ The percentage of correction ranked in top 'n' among all errors in
        this dataset.
        
        If there exists error object with no candidate in this dataset, this
        function counts such object as an 'out of top' instance.
        
        Args:
            n: a integer indicates the number of top candidates are used to
                search a correction. (default: infinity)

        Returns:
            The precision of the correction in top 'n' candidates. If the value
            of 'n' is infinity, returns the percentage of correction ever exists
            as candidates.
        
        Raises:
            ValueError: If 'n' is not positive.
            ConfidenceUnsetError: If any error confidence is unset.
            ConfidenceRankError: If there is not error objects in this dataset.
        """
        if len(self.errors) == 0:
            raise ConfidenceRankError
        if n <= 0:
            raise ValueError
        elif n == float('inf'):
            def exists_true_label(error):
                for c in error.candidates:
                    if c.label == True:
                        return True
                return False
            count = sum([1 if exists_true_label(e) else 0 for e in self.errors])
        else:
            def get_rank_savely(err):
                try:
                    return err.rank
                except ConfidenceRankError:
                    return float('inf')
            count = sum([1 if get_rank_savely(e) <= n else 0 for e in self.errors])
        return count / float(len(self.errors))
        
