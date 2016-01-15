import unittest

from autosklearn.pipeline.components.classification.feed_nn import Feed_NN
from autosklearn.pipeline.util import _test_classifier, _test_classifier_predict_proba

import sklearn.metrics
import numpy as np


class FeedForwardComponentTest(unittest.TestCase):
    def test_default_configuration(self):
        # What about random_state?
        for i in range(10):
            predictions, targets = _test_classifier(Feed_NN, dataset='iris')
            print sklearn.metrics.accuracy_score(predictions, targets)
            # self.assertAlmostEqual(0.96,
            #                       sklearn.metrics.accuracy_score(predictions, targets))

    def test_default_configuration_space(self):
        for i in range(10):
            # Simple test that hopefully works
            dataset_dir = '../../data/'
            # Take training and data from binary classification
            X_train = np.load(dataset_dir + 'unit_train.npy')
            y_train = np.load(dataset_dir + 'unit_train_labels.npy')
            X_test = np.load(dataset_dir + 'unit_test.npy')
            y_test = np.load(dataset_dir + 'unit_test_labels.npy')

            configuration_space = Feed_NN.get_hyperparameter_search_space()
            default = configuration_space.get_default_configuration()

            cls = Feed_NN(**{hp_name: default[hp_name] for hp_name in
                               default if default[hp_name] is not None})

            cls = cls.fit(X_train, y_train)
            prediction = cls.predict_proba(X_test)
            print sklearn.metrics.log_loss(y_test, prediction)
            # TODO: Ask Matthias about the value
            # self.assertAlmostEqual(sklearn.metrics.log_loss(y_test, prediction),
            #                       0.68661222917147913)
