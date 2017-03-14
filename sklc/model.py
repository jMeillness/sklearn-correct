import glob
import logging
import numpy as np
import operator
import os
from os import path
import re
from sklearn import base
from sklearn import cross_validation
from sklearn import ensemble
from sklearn import grid_search
from sklearn import kernel_ridge
from sklearn import neural_network
from sklearn import svm
from sklearn.externals import joblib


class ModelError(Exception):
    pass


class SKLModel(object):
    """ A learnig model for errors correction.
        
    Attributes:
        _gs: an object that implements the 'sklearn.grid_search.GridSearchCV'
            interface.
        settings: the settings of the model.
    """

    def __init__(self, estimator, param_grid, train_data, **kwargs):
        """ Initialize a trained scikit-learn model.
            
        Args:
            estimator: an object that implements the scikit-learn estimator
                interface.
            param_grid: a dictionary used as 'param_grid' for cross-validation
                in scikit-learn.
            train_data: a dataset used for training.
            
        Kwargs:
            pkl_path: a string represents the pathname to the pre-trained model
                file.
            override: a boolean flag forces the model retraining even the
                pre-trained model file is provided. (default: False)
            weighted: a boolean flag indicates whether the training sample
                should be weighted. (default: False)
            cv: the number of folds used in cross validataion. (default: 10)
            load: a boolean flag indicates whether initializing from data file.
                (default: False)
            
        Raises:
            ValueError: An invalid scikit-learn estimator given in parameter.
        """
        override = kwargs.get('override', False)
        weighted = kwargs.get('weighted', False)
        cv = kwargs.get('cv', 10)
        pkl_path = kwargs.get('pkl_path', None)
        load = kwargs.get('load', False)

        print(estimator.get_params().keys())
        
        if load:
            return
        
        if not override and pkl_path != None and len(glob.glob(pkl_path + '*')) > 0:
            # Load a pre-trained model, if a valid pathname is given and no
            # force retraining.
            logging.debug('load pre-trained model form file %s' % pkl_path)
            self._gs = joblib.load(pkl_path) 
        else:
            # Train a model.
            logging.debug('training model using grid search...')
            if weighted:
                fit_weights={'sample_weight': np.array(train_data.feature_weights)}
            else:
                fit_weights={}
            self._gs = grid_search.GridSearchCV( 
                    estimator=estimator,
                    param_grid=param_grid,
                    fit_params=fit_weights,
                    n_jobs=-1,
                    cv=cv,
                    )
            self._gs.fit(np.array(train_data.feature_values), np.array(train_data.labels))
            if pkl_path != None:
                self.write(pkl_path)

    @property
    def settings(self):
        return self._gs.best_params_

    def predict(self, test_data):
        """ Make predition to the test data.
            
        Args:
            test_data: a string represents the pathname to the pre-trained model
                file.
        """
        confidences = self._gs.best_estimator_ \
                .predict(np.array(test_data.feature_values)) \
                .reshape(-1,).tolist()
        test_data.confidences = confidences
        return confidences

    @staticmethod
    def load(pathname):
        """ Load a pre-trained model from file.
            
        Args:
            pathname: a string represents the pathname to the pre-trained model
                file.
         
        Returns:
            An object of this class.
            
        Raises:
            ValueError: An error occurred accessing the given pathname.
        """
        if pathname == None or len(glob.glob(pathname + '*')) == 0:
            raise ValueError('%s is not a valid path.' % pathname)
        model = SKLModel(base.BaseEstimator(), None, None, load=True)
        model._gs = joblib.load(pathname) 
        return model

    def write(self, pathname):
        """ Write this model to file. This model can be reconstruct using 
            
        Args:
            pathname: a string represents the pathname to the pre-trained model
                file.
        """
        dirname = path.dirname(pathname)
        if not path.exists(dirname):
            os.makedirs(dirname)
        joblib.dump(self._gs, pathname) 


class RandomForestModel(SKLModel):

    DEFAULT_PARAM_GRID = dict(
            n_estimators = np.arange(50, 500, 50),
            min_samples_split = np.arange(2, 10, 2),
            )
    ESTIMATOR = ensemble.RandomForestRegressor()

    def __init__(self, train_data, **kwargs):
        """
        Kwargs:
            param_grid: a parameter grid used for training.
        """
        param_grid = kwargs.pop('param_grid',
                RandomForestModel.DEFAULT_PARAM_GRID)
        super(RandomForestModel, self).__init__( \
                RandomForestModel.ESTIMATOR,
                param_grid,
                train_data,
                **kwargs
                )


class KernelRidgeModel(SKLModel):

    DEFAULT_PARAM_GRID = dict(
            alpha = [1e0],
            gamma = [1e0],
            )
    ESTIMATOR = kernel_ridge.KernelRidge(kernel='rbf', gamma=0.1)

    def __init__(self, train_data, **kwargs):
        """
        Kwargs:
            param_grid: a parameter grid used for training.
        """
        param_grid = kwargs.pop('param_grid',
                KernelRidgeModel.DEFAULT_PARAM_GRID)
        super(KernelRidgeModel, self).__init__( \
                KernelRidgeModel.ESTIMATOR,
                param_grid,
                train_data,
                **kwargs
                )


class ExtraTreesModel(SKLModel):

    DEFAULT_PARAM_GRID = dict(
            n_estimators = np.arange(10, 200, 10),
            min_samples_split = np.arange(2, 10, 1),
            bootstrap=[True],
            )
    ESTIMATOR = ensemble.ExtraTreesRegressor()

    def __init__(self, train_data, **kwargs):
        """
        Kwargs:
            param_grid: a parameter grid used for training.
        """
        param_grid = kwargs.pop('param_grid',
                ExtraTreesModel.DEFAULT_PARAM_GRID)
        super(ExtraTreesModel, self).__init__( \
                ExtraTreesModel.ESTIMATOR,
                param_grid,
                train_data,
                **kwargs
                )


class AdaBoostModel(SKLModel):

    DEFAULT_PARAM_GRID = dict(
            n_estimators = np.arange(50, 500, 50),
            loss=['linear', 'square', 'exponential']
            )
    ESTIMATOR = ensemble.AdaBoostRegressor()

    def __init__(self, train_data, **kwargs):
        """
        Kwargs:
            param_grid: a parameter grid used for training.
        """
        param_grid = kwargs.pop('param_grid',
                AdaBoostModel.DEFAULT_PARAM_GRID)
        super(AdaBoostModel, self).__init__( \
                AdaBoostModel.ESTIMATOR,
                param_grid,
                train_data,
                **kwargs
                )


class GradientBoostingModel(SKLModel):

    DEFAULT_PARAM_GRID = dict(
            loss = ['ls', 'lad', 'huber', 'quantile'],
            n_estimators = np.arange(100, 500, 100),
            min_samples_split = np.arange(2, 5, 1),
            max_depth = np.arange(1, 5, 1),
            )
    ESTIMATOR = ensemble.GradientBoostingRegressor()

    def __init__(self, train_data, **kwargs):
        """
        Kwargs:
            param_grid: a parameter grid used for training.
        """
        param_grid = kwargs.pop('param_grid',
                GradientBoostingModel.DEFAULT_PARAM_GRID)
        super(GradientBoostingModel, self).__init__( \
                GradientBoostingModel.ESTIMATOR,
                param_grid,
                train_data,
                **kwargs
                )


class SupportVectorModel(SKLModel):

    DEFAULT_PARAM_GRID = dict(
            kernel = ['rbf'],
            C = [1e0],
            gamma = np.logspace(-2, 2, 5),
            )
    ESTIMATOR = svm.SVR()

    def __init__(self, train_data, **kwargs):
        """
        Kwargs:
            param_grid: a parameter grid used for training.
        """
        param_grid = kwargs.pop('param_grid',
                SupportVectorModel.DEFAULT_PARAM_GRID)
        super(SupportVectorModel, self).__init__( \
                SupportVectorModel.ESTIMATOR,
                param_grid,
                train_data,
                **kwargs
                )


class MLPModel(SKLModel):

    DEFAULT_PARAM_GRID = dict(
            activation = ['logistic', 'tanh', 'relu'],
            algorithm = ['sgd'],
            )
    ESTIMATOR = neural_network.MLPRegressor()

    def __init__(self, train_data, **kwargs):
        """
        Kwargs:
            param_grid: a parameter grid used for training.
        """
        param_grid = kwargs.pop('param_grid',
                MLPModel.DEFAULT_PARAM_GRID)
        super(MLPModel, self).__init__( \
                MLPModel.ESTIMATOR,
                param_grid,
                train_data,
                **kwargs
                )
