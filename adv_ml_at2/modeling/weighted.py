from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils.validation import has_fit_parameter


class TailWeightedEstimator(RegressorMixin, BaseEstimator):
    """Wrap a scikit-learn regressor with training-only tail weighting."""

    def __init__(self, estimator, tail_weight_multiplier=1.0, lower_quantile=0.10, upper_quantile=0.90, target_divisor=1.0):
        self.estimator = estimator
        self.tail_weight_multiplier = tail_weight_multiplier
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.target_divisor = target_divisor

    def fit(self, X, y):
        if self.tail_weight_multiplier < 1:
            raise ValueError("tail_weight_multiplier must be at least 1.")
        if self.target_divisor <= 0:
            raise ValueError("target_divisor must be greater than zero.")
        y_array = np.asarray(y, dtype=float).reshape(-1)
        self.lower_threshold_ = float(np.quantile(y_array, self.lower_quantile))
        self.upper_threshold_ = float(np.quantile(y_array, self.upper_quantile))
        tail_mask = (y_array <= self.lower_threshold_) | (y_array >= self.upper_threshold_)
        weights = np.where(tail_mask, float(self.tail_weight_multiplier), 1.0)
        self.estimator_ = clone(self.estimator)
        y_scaled = y_array / float(self.target_divisor)
        if has_fit_parameter(self.estimator_, "sample_weight"):
            self.estimator_.fit(X, y_scaled, sample_weight=weights)
            self.resampled_training_rows_ = len(y_array)
        else:
            if int(self.tail_weight_multiplier) != self.tail_weight_multiplier:
                raise ValueError("tail_weight_multiplier must be an integer when the estimator does not accept sample_weight.")
            X_array = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
            repetitions = np.where(tail_mask, int(self.tail_weight_multiplier), 1)
            X_resampled = np.repeat(X_array, repetitions, axis=0)
            if hasattr(X, "columns"):
                X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
            self.estimator_.fit(X_resampled, np.repeat(y_scaled, repetitions))
            self.resampled_training_rows_ = int(np.sum(repetitions))
        self.n_features_in_ = self.estimator_.n_features_in_
        self.original_training_rows_ = len(y_array)
        self.tail_training_rows_ = int(tail_mask.sum())
        for attribute in ("coef_", "intercept_", "feature_importances_", "n_iter_", "loss_curve_"):
            if hasattr(self.estimator_, attribute):
                setattr(self, attribute, getattr(self.estimator_, attribute))
        if hasattr(self, "n_iter_") and hasattr(self.estimator_, "max_iter"):
            self.converged_ = self.n_iter_ < self.estimator_.max_iter
        if hasattr(self, "loss_curve_"):
            self.initial_scaled_loss_ = float(self.loss_curve_[0])
            self.final_scaled_loss_ = float(self.loss_curve_[-1])
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def predict(self, X):
        return np.asarray(self.estimator_.predict(X)).reshape(-1) * float(self.target_divisor)


def make_tail_weighted_estimator(estimator, tail_weight_multiplier=1.0, lower_quantile=0.10, upper_quantile=0.90, target_divisor=1.0):
    """Return a fitted-by-scikit-learn wrapper for any regressor."""
    return TailWeightedEstimator(estimator, tail_weight_multiplier, lower_quantile, upper_quantile, target_divisor)


def tail_mask_from_estimator(estimator, y):
    y_array = np.asarray(y)
    return (y_array <= estimator.lower_threshold_) | (y_array >= estimator.upper_threshold_)


def negative_rmse(estimator, X, y):
    from sklearn.metrics import root_mean_squared_error
    return -root_mean_squared_error(y, estimator.predict(X))


def negative_mae(estimator, X, y):
    from sklearn.metrics import mean_absolute_error
    return -mean_absolute_error(y, estimator.predict(X))


def negative_tail_rmse(estimator, X, y):
    from sklearn.metrics import root_mean_squared_error
    mask = tail_mask_from_estimator(estimator, y)
    return -root_mean_squared_error(np.asarray(y)[mask], estimator.predict(X)[mask])


def negative_tail_mae(estimator, X, y):
    from sklearn.metrics import mean_absolute_error
    mask = tail_mask_from_estimator(estimator, y)
    return -mean_absolute_error(np.asarray(y)[mask], estimator.predict(X)[mask])
