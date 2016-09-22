from .. import data
from .. import model
import numpy as np
from os import path
import shutil
import testutils
import unittest


class SKLModelTest(unittest.TestCase):

    def setUp(self):
        self.data_path = testutils.get_resource_file('data.sample.txt')
        self.dataset = data.Dataset.read(self.data_path)
        self.train_dataset = self.dataset.subset(
                filt=lambda x: x.name == 'Error1')
        self.test_dataset = self.dataset.subset(
                filt=lambda x: x.name != 'Error1')
        self.write_path = 'tmp/model/test.model'
        self.model_kwargs = dict(pkl_path=self.write_path, cv=3)
        self.model_weighted_kwargs = dict(weighted=True, cv=3)

    def tearDown(self):
        if path.exists('tmp'):
            shutil.rmtree('tmp')

    def test_init_from_file(self):
        new_model = model.SKLModel(None, None, None,
                pkl_path=self.write_path)
        self.assertEqual(self.model.settings, new_model.settings)

    def test_load(self):
        new_model = model.SKLModel.load(self.write_path)
        self.assertEqual(self.model.settings, new_model.settings)

    def test_load_error(self):
        self.assertRaises(ValueError, model.SKLModel.load,
                self.write_path + '.notexist')

    def test_write_and_load(self):
        self.model.write(self.write_path)
        new_model = model.SKLModel.load(self.write_path)
        self.assertEqual(self.model.settings, new_model.settings)

    def test_predict(self):
        self.model.predict(self.test_dataset)


class RandomForestModelTest(SKLModelTest):

    def setUp(self):
        super(RandomForestModelTest, self).setUp()
        param_grid=dict(n_estimators = np.arange(10, 20, 10)),
        self.model = model.RandomForestModel(self.train_dataset,
                param_grid=param_grid, **self.model_kwargs)
        self.model_weighted = model.RandomForestModel(self.train_dataset,
                param_grid=param_grid, **self.model_weighted_kwargs)


class KernelRidgeModelTest(SKLModelTest):

    def setUp(self):
        super(KernelRidgeModelTest, self).setUp()
        param_grid=dict(),
        self.model = model.KernelRidgeModel(self.train_dataset,
                param_grid=param_grid, **self.model_kwargs)
        self.model_weighted = model.KernelRidgeModel(self.train_dataset,
                param_grid=param_grid, **self.model_weighted_kwargs)


class ExtraTreesModelTest(SKLModelTest):

    def setUp(self):
        super(ExtraTreesModelTest, self).setUp()
        param_grid=dict(min_samples_split = np.arange(1, 2, 1)),
        self.model = model.ExtraTreesModel(self.train_dataset,
                param_grid=param_grid, **self.model_kwargs)
        self.model_weighted = model.ExtraTreesModel(self.train_dataset,
                param_grid=param_grid, **self.model_weighted_kwargs)


class AdaBoostModelTest(SKLModelTest):

    def setUp(self):
        super(AdaBoostModelTest, self).setUp()
        param_grid=dict(loss=['linear', 'square']),
        self.model = model.AdaBoostModel(self.train_dataset,
                param_grid=param_grid, **self.model_kwargs)
        self.model_weighted = model.AdaBoostModel(self.train_dataset,
                param_grid=param_grid, **self.model_weighted_kwargs)


class GradientBoostingModelTest(SKLModelTest):

    def setUp(self):
        super(GradientBoostingModelTest, self).setUp()
        param_grid=dict(loss=['ls', 'lad']),
        self.model = model.GradientBoostingModel(self.train_dataset,
                param_grid=param_grid, **self.model_kwargs)
        self.model_weighted = model.GradientBoostingModel(self.train_dataset,
                param_grid=param_grid, **self.model_weighted_kwargs)


class SupportVectorModelTest(SKLModelTest):

    def setUp(self):
        super(SupportVectorModelTest, self).setUp()
        param_grid=dict(C = [1e0]),
        self.model = model.SupportVectorModel(self.train_dataset,
                param_grid=param_grid, **self.model_kwargs)
        self.model_weighted = model.SupportVectorModel(self.train_dataset,
                param_grid=param_grid, **self.model_weighted_kwargs)


class MLPModelTest(SKLModelTest):

    def setUp(self):
        super(MLPModelTest, self).setUp()
        param_grid=dict(activation = ['logistic']),
        self.model = model.MLPModel(self.train_dataset,
                param_grid=param_grid, **self.model_kwargs)
        self.model_weighted = model.MLPModel(self.train_dataset,
                param_grid=param_grid, **self.model_weighted_kwargs)
