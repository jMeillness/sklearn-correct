from .. import data
import math
import testutils
import unittest


class FeatureTest(unittest.TestCase):

    def setUp(self):
        self.names = testutils.random_strings(10)

    def test_name(self):
        for name in self.names:
            name = 'abc'
            feat = data.Feature('abc')
            self.assertEqual(name, feat.name)
            self.assertEqual(name, str(feat))


class FeatureRegistryTest(unittest.TestCase):

    def setUp(self):
        self.feature_registry = data.FeatureRegistry()
        self.names = testutils.random_strings(10)
        self.features = [data.Feature(name) for name in self.names]
        for i in range(len(self.features)):
            self.feature_registry.register(self.features[i])

    def test_get(self):
        for i in range(len(self.features)):
            self.assertEqual(self.features[i], self.feature_registry.get(i))
            self.assertEqual(i, self.feature_registry.get(self.features[i]))
            self.assertEqual(i, self.feature_registry.get(self.names[i]))

    def test_register_exist_feature(self):
        for feat in self.features:
            self.assertRaises(ValueError, self.feature_registry.register, feat)

    def test_get_non_exist_key(self):
        import random
        import sys
        while True:
            id_non_exist = random.randint(- sys.maxint - 1, sys.maxint)
            if id_non_exist not in range(len(self.features)):
                break
        self.assertRaises(KeyError, self.feature_registry.get, id_non_exist)
        while True:
            name_non_exist = testutils.random_string()
            if name_non_exist not in self.names:
                break
        self.assertRaises(KeyError, self.feature_registry.get, name_non_exist)
        feat_non_exist = data.Feature(name_non_exist)
        self.assertRaises(KeyError, self.feature_registry.get, feat_non_exist)

    def test_get_type_error(self):
        self.assertRaises(TypeError, self.feature_registry.get, object())


class ErrorTest(unittest.TestCase):

    def setUp(self):
        self.data_file = testutils.get_resource_file('data.sample.txt')
        self.dataset = data.Dataset.read(self.data_file)

    def test_feature_values(self):
        self.assertEqual([
            [0.1, 0.2, 0.3],
            [0.2, 0.0, 0.8],
            [0.3, 0.4, 0.1],
            ], self.dataset.errors[0].feature_values)
        self.assertEqual([
            [0.1, 0.2, 0.3],
            [0.5, 0.4, 0.1],
            ], self.dataset.errors[1].feature_values)

    def test_feature_weights(self):
        self.assertEqual([0.5, 1, 0.5], self.dataset.errors[0].feature_weights)
        self.assertEqual([1, 1], self.dataset.errors[1].feature_weights)

    def test_labels(self):
        self.assertEqual([1, 0, 1], self.dataset.errors[0].labels)
        self.assertEqual([0, 1], self.dataset.errors[1].labels)

    def test_confidences(self):
        error1 = self.dataset.errors[0]
        self.assertEqual([-1, -1, -1], error1.confidences)
        error1.confidences = [0.1, 0.2, 0.3]
        self.assertEqual([0.1, 0.2, 0.3], error1.confidences)

    def test_confidences_set_error(self):
        def set_error_case():
            self.dataset.errors[0].confidences = [0.1, 0.2]
        self.assertRaises(ValueError, set_error_case)

    def test_rank(self):
        error = self.dataset.errors[0]
        error.confidences = [0.2, 0.1, 0.3]
        self.assertEqual(1, error.rank)
        error.confidences = [0.1, 0.3, 0.2]
        self.assertEqual(2, error.rank)
        error.confidences = [0.1, 0.3, 0.3]
        self.assertEqual(1, error.rank)
        error.confidences = [0.1, 0.1, 0.1]
        self.assertEqual(1, error.rank)

    def test_rank_confidence_unset_error(self):
        def call_rank():
            self.dataset.errors[0].rank
        self.assertRaises(data.ConfidenceUnsetError, call_rank)

    def test_rank_confidence_rank_error(self):
        def call_rank():
            self.dataset.errors[0].candidates = []
            self.dataset.errors[0].rank
        self.assertRaises(data.ConfidenceRankError, call_rank)


class DatasetTest(unittest.TestCase):

    def setUp(self):
        self.data_file = testutils.get_resource_file('data.sample.txt')
        self.dataset = data.Dataset.read(self.data_file)
    
    def test_read(self):
        self.assertEqual('Feature1', self.dataset.feature_registry.get(0).name)
        self.assertEqual('Feature2', self.dataset.feature_registry.get(1).name)
        self.assertEqual('Feature3', self.dataset.feature_registry.get(2).name)
        error1 = self.dataset.errors[0]
        self.assertEqual('Error1', error1.name)
        err1c1 = error1.candidates[0]
        self.assertEqual('Candidate1', err1c1.name)
        self.assertEqual(0.1, err1c1.feature_values[0])
        self.assertEqual(0.2, err1c1.feature_values[1])
        self.assertEqual(0.3, err1c1.feature_values[2])
        self.assertEqual(1, err1c1.label)
        err1c2 = error1.candidates[1]
        self.assertEqual('Candidate2', err1c2.name)
        self.assertEqual(0.2, err1c2.feature_values[0])
        self.assertEqual(0.0, err1c2.feature_values[1])
        self.assertEqual(0.8, err1c2.feature_values[2])
        self.assertEqual(0, err1c2.label)
        err1c3 = error1.candidates[2]
        self.assertEqual('Candidate3', err1c3.name)
        self.assertEqual(0.3, err1c3.feature_values[0])
        self.assertEqual(0.4, err1c3.feature_values[1])
        self.assertEqual(0.1, err1c3.feature_values[2])
        self.assertEqual(1, err1c3.label)
        error2 = self.dataset.errors[1]
        self.assertEqual('Error2', error2.name)
        err2c1 = error2.candidates[0]
        self.assertEqual('Candidate4', err2c1.name)
        self.assertEqual(0.1, err2c1.feature_values[0])
        self.assertEqual(0.2, err2c1.feature_values[1])
        self.assertEqual(0.3, err2c1.feature_values[2])
        self.assertEqual(0, err2c1.label)
        err2c2 = error2.candidates[1]
        self.assertEqual('Candidate5', err2c2.name)
        self.assertEqual(0.5, err2c2.feature_values[0])
        self.assertEqual(0.4, err2c2.feature_values[1])
        self.assertEqual(0.1, err2c2.feature_values[2])
        self.assertEqual(1, err2c2.label)

    def test_feature_values(self):
        self.assertEqual([
            [0.1, 0.2, 0.3],
            [0.2, 0.0, 0.8],
            [0.3, 0.4, 0.1],
            [0.1, 0.2, 0.3],
            [0.5, 0.4, 0.1],
            ], self.dataset.feature_values)

    def test_labels(self):
        self.assertEqual([1, 0, 1, 0, 1], self.dataset.labels)

    def test_subset_with_filter(self):
        sub = self.dataset.subset(filt=lambda x: True)
        self.assertEqual(self.dataset.errors, sub.errors)

        sub, complement = self.dataset.subset(
                    filt=lambda x: False,
                    return_complement=True
                    )
        self.assertEqual([], sub.errors)
        self.assertEqual(self.dataset.errors, complement.errors)

        sub, complement = self.dataset.subset(
                    filt=lambda x: x.name == 'Error1',
                    return_complement=True
                    )
        self.assertEqual([self.dataset.errors[0]], sub.errors)
        self.assertEqual([self.dataset.errors[1]], complement.errors)

        sub, complement = self.dataset.subset(
                    filt=lambda x: x.name == 'Error2',
                    return_complement=True
                    )
        self.assertEqual([self.dataset.errors[1]], sub.errors)
        self.assertEqual([self.dataset.errors[0]], complement.errors)

    def test_subset_with_first(self):
        sub = self.dataset.subset(first=2)
        self.assertEqual(self.dataset.errors, sub.errors)

        sub, complement = self.dataset.subset(
                    first=1,
                    return_complement=True
                    )
        self.assertEqual([self.dataset.errors[0]], sub.errors)
        self.assertEqual([self.dataset.errors[1]], complement.errors)

        sub, complement = self.dataset.subset(
                    first=0,
                    return_complement=True
                    )
        self.assertEqual([], sub.errors)
        self.assertEqual(self.dataset.errors, complement.errors)

        sub, complement = self.dataset.subset(
                    first=0.3,
                    return_complement=True
                    )
        self.assertEqual([], sub.errors)
        self.assertEqual(self.dataset.errors, complement.errors)

        sub, complement = self.dataset.subset(
                    first=0.5,
                    return_complement=True
                    )
        self.assertEqual([self.dataset.errors[0]], sub.errors)
        self.assertEqual([self.dataset.errors[1]], complement.errors)

        sub, complement = self.dataset.subset(
                    first=0.7,
                    return_complement=True
                    )
        self.assertEqual([self.dataset.errors[0]], sub.errors)
        self.assertEqual([self.dataset.errors[1]], complement.errors)

    def test_subset_with_last(self):
        sub = self.dataset.subset(last=2)
        self.assertEqual(self.dataset.errors, sub.errors)

        sub, complement = self.dataset.subset(
                    last=1,
                    return_complement=True
                    )
        self.assertEqual([self.dataset.errors[1]], sub.errors)
        self.assertEqual([self.dataset.errors[0]], complement.errors)

        sub, complement = self.dataset.subset(
                    last=0,
                    return_complement=True
                    )
        self.assertEqual([], sub.errors)
        self.assertEqual(self.dataset.errors, complement.errors)

        sub, complement = self.dataset.subset(
                    last=0.3,
                    return_complement=True
                    )
        self.assertEqual([], sub.errors)
        self.assertEqual(self.dataset.errors, complement.errors)

        sub, complement = self.dataset.subset(
                    last=0.5,
                    return_complement=True
                    )
        self.assertEqual([self.dataset.errors[1]], sub.errors)
        self.assertEqual([self.dataset.errors[0]], complement.errors)

        sub, complement = self.dataset.subset(
                    last=0.7,
                    return_complement=True
                    )
        self.assertEqual([self.dataset.errors[1]], sub.errors)
        self.assertEqual([self.dataset.errors[0]], complement.errors)

    def test_subset_value_error(self):
        self.assertRaises(ValueError, self.dataset.subset, **dict())
        self.assertRaises(ValueError, self.dataset.subset,
                **dict(first=1, last=0.3))

    def test_confidences(self):
        error1 = self.dataset
        self.assertEqual([-1, -1, -1, -1, -1], error1.confidences)
        error1.confidences = [0.1, 0.2, 0.3, 0.4, 0.5]
        self.assertEqual([0.1, 0.2, 0.3, 0.4, 0.5], error1.confidences)

    def test_confidences_set_error(self):
        def set_error_case():
            self.dataset.confidences = [0.1, 0.2]
        self.assertRaises(ValueError, set_error_case)

    def test_feature_weights(self):
        w = 2.0 / 3
        self.assertEqual([w, 1, w, 1, w], self.dataset.feature_weights)

    def test_precision(self):
        self.dataset.errors[0].confidences = [0.2, 0.1, 0.3]
        self.dataset.errors[1].confidences = [0.1, 0.1]
        self.assertEqual(1, self.dataset.precision_at(1))
        self.assertEqual(1, self.dataset.precision_at(2))
        self.assertEqual(1, self.dataset.precision_at(10))
        self.assertEqual(1, self.dataset.precision_at())
        self.dataset.errors[1].confidences = [0.2, 0.1]
        self.assertEqual(0.5, self.dataset.precision_at(1))
        self.assertEqual(1, self.dataset.precision_at(2))
        self.assertEqual(1, self.dataset.precision_at(10))
        self.assertEqual(1, self.dataset.precision_at())
        self.dataset.errors[1].candidates = []
        self.assertEqual(0.5, self.dataset.precision_at(1))
        self.assertEqual(0.5, self.dataset.precision_at(10))
        self.assertEqual(0.5, self.dataset.precision_at())

    def test_precision_value_error(self):
        self.assertRaises(ValueError, self.dataset.precision_at, 0)

    def test_precision_confidence_unset_error(self):
        self.assertRaises(data.ConfidenceUnsetError,
                self.dataset.precision_at, 1)

    def test_precision_confidence_rank_error(self):
        self.dataset.errors = []
        self.assertRaises(data.ConfidenceRankError,
                self.dataset.precision_at, 1)
