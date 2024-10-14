import numpy as np
from numpy.core.umath_tests import inner1d
from copy import deepcopy
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import OneHotEncoder

class AdaBoost_CNN:
    def __init__(self, base_estimator, n_estimators=50, learning_rate=1, algorithm='SAMME.R', random_state=None, epochs=6):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.algorithm = algorithm
        self.random_state = random_state
        self.epochs = epochs
        self.estimators_ = []
        self.estimator_weights_ = np.zeros(self.n_estimators)
        self.estimator_errors_ = np.ones(self.n_estimators)

    def _samme_proba(self, estimator, n_classes, X):
        proba = estimator.predict(X)
        proba[proba < np.finfo(proba.dtype).eps] = np.finfo(proba.dtype).eps
        log_proba = np.log(proba)
        return (n_classes - 1) * (log_proba - (1. / n_classes) * log_proba.sum(axis=1)[:, np.newaxis])

    def fit(self, X, y, batch_size):
        self.batch_size = batch_size
        self.n_samples = X.shape[0]
        self.classes_ = np.array(sorted(list(set(y))))
        self.n_classes_ = len(self.classes_)

        sample_weight = np.ones(self.n_samples) / self.n_samples

        for iboost in range(self.n_estimators):
            sample_weight, estimator_weight, estimator_error = self.boost(X, y, sample_weight)
            if estimator_error is None:
                break
            self.estimator_errors_[iboost] = estimator_error
            self.estimator_weights_[iboost] = estimator_weight
            if estimator_error <= 0:
                break

        return self

    def boost(self, X, y, sample_weight):
        if self.algorithm == 'SAMME':
            return self.discrete_boost(X, y, sample_weight)
        return self.real_boost(X, y, sample_weight)

    def real_boost(self, X, y, sample_weight):
        estimator = self.deepcopy_CNN(self.base_estimator if not self.estimators_ else self.estimators_[-1])
        y_b = OneHotEncoder(sparse_output=False).fit_transform(y.reshape(-1, 1))
        estimator.fit(X, y_b, sample_weight=sample_weight, epochs=self.epochs, batch_size=self.batch_size)
        y_pred_l = np.argmax(estimator.predict(X), axis=1)
        incorrect = y_pred_l != y
        estimator_error = np.dot(incorrect, sample_weight) / np.sum(sample_weight)

        if estimator_error >= 1 - 1 / self.n_classes_:
            return None, None, None

        y_predict_proba = estimator.predict(X)
        y_predict_proba[y_predict_proba < np.finfo(y_predict_proba.dtype).eps] = np.finfo(y_predict_proba.dtype).eps
        y_coding = np.array([-1. / (self.n_classes_ - 1), 1.]).take(self.classes_ == y[:, np.newaxis])

        sample_weight *= np.exp(-1. * self.learning_rate * ((self.n_classes_ - 1) / self.n_classes_) * inner1d(y_coding, np.log(y_predict_proba)))
        sample_weight /= np.sum(sample_weight)
        self.estimators_.append(estimator)

        return sample_weight, 1, estimator_error

    def discrete_boost(self, X, y, sample_weight):
        estimator = self.deepcopy_CNN(self.base_estimator if not self.estimators_ else self.estimators_[-1])
        y_b = OneHotEncoder(sparse_output=False).fit_transform(y.reshape(-1, 1))
        estimator.fit(X, y_b, sample_weight=sample_weight, epochs=self.epochs, batch_size=self.batch_size)
        y_pred_l = np.argmax(estimator.predict(X), axis=1)
        incorrect = y_pred_l != y
        estimator_error = np.dot(incorrect, sample_weight) / np.sum(sample_weight)

        if estimator_error >= 1 - 1 / self.n_classes_:
            return None, None, None

        estimator_weight = self.learning_rate * np.log((1. - estimator_error) / estimator_error + np.log(self.n_classes_ - 1))
        sample_weight *= np.exp(estimator_weight * incorrect)
        sample_weight /= np.sum(sample_weight)
        self.estimators_.append(estimator)

        return sample_weight, estimator_weight, estimator_error

    def predict(self, X):
        classes = self.classes_[:, np.newaxis]
        if self.algorithm == 'SAMME.R':
            pred = sum(self._samme_proba(estimator, self.n_classes_, X) for estimator in self.estimators_)
        else:
            pred = sum((estimator.predict(X).argmax(axis=1) == classes).T * w for estimator, w in zip(self.estimators_, self.estimator_weights_))
        pred /= self.estimator_weights_.sum()
        return self.classes_.take(np.argmax(pred, axis=1), axis=0)

    def deepcopy_CNN(self, base_estimator):
        config = base_estimator.get_config()
        estimator = Sequential.from_config(config)
        estimator.set_weights(base_estimator.get_weights())
        estimator.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return estimator
