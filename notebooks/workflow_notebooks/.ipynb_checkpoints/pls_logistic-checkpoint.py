from sklearn.cross_decomposition import PLSRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
import numpy as np
from sklearn import datasets
from sklearn.metrics import (brier_score_loss, precision_score, recall_score,
                             f1_score)

def logistic_fx(value, x0=0.5, scale=0.5):
    """ Function parameters were found through manual inspection reference Notebook"""
    import math
    numerator = math.e**((x0-value)/scale)
    denomenator = 1 + numerator
    return numerator/denomenator

logistic_fx = np.vectorize(logistic_fx)

class PLSLogistic(PLSRegression):
    """
    Class to inherit SciKit-Learn classifier
    """
    def __init__(self, n_components=2, logistic_scaler=0.5):
        self.logistic_scaler = 0.5
        PLSRegression.__init__(self, n_components=n_components)

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        super().fit(X, y)

        return self


    def predict_proba(self, X):
        regression_log = logistic_fx(super().predict(X), scale=self.logistic_scaler)
        regression_log = super().predict(X)
        probabilities = np.concatenate([1 - regression_log, regression_log], axis=1)
        return probabilities

    def predict(self, X):
        probabilities = self.predict_proba(X)

        classifications = probabilities.copy()
        classifications[probabilities >= 0.5] = 1
        classifications[probabilities < 0.5] = 0
        return classifications



X, y = datasets.make_classification(n_samples=10000, n_features=20,
                                    n_informative=2, n_redundant=10,
                                    random_state=42)

pls_log = PLSLogistic()

isotonic = CalibratedClassifierCV(pls_log, cv=2, method='isotonic')
# pls_log.fit(X, y)
isotonic.fit(X, y)

prob_pos = isotonic.predict_proba(X)[:, 1]

calibration_curve(y, prob_pos, n_bins=10)

clf_score = brier_score_loss(y, prob_pos, pos_label=y.max())