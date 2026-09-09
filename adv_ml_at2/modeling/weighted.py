from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.neural_network import MLPRegressor
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
        else:
            if int(self.tail_weight_multiplier) != self.tail_weight_multiplier:
                raise ValueError("tail_weight_multiplier must be an integer when the estimator does not accept sample_weight.")
            X_array = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
            repetitions = np.where(tail_mask, int(self.tail_weight_multiplier), 1)
            X_resampled = np.repeat(X_array, repetitions, axis=0)
            if hasattr(X, "columns"):
                X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
            self.estimator_.fit(X_resampled, np.repeat(y_scaled, repetitions))
        self.n_features_in_ = self.estimator_.n_features_in_
        self.original_training_rows_ = len(y_array)
        self.tail_training_rows_ = int(tail_mask.sum())
        self.resampled_training_rows_ = int(np.sum(np.where(tail_mask, int(self.tail_weight_multiplier), 1)))
        for attribute in ("coef_", "intercept_", "feature_importances_", "n_iter_", "loss_curve_"):
            if hasattr(self.estimator_, attribute):
                setattr(self, attribute, getattr(self.estimator_, attribute))
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def predict(self, X):
        return np.asarray(self.estimator_.predict(X)).reshape(-1) * float(self.target_divisor)


def make_tail_weighted_estimator(estimator, tail_weight_multiplier=1.0, lower_quantile=0.10, upper_quantile=0.90, target_divisor=1.0):
    """Return a fitted-by-scikit-learn wrapper for any regressor."""
    return TailWeightedEstimator(estimator, tail_weight_multiplier, lower_quantile, upper_quantile, target_divisor)


class TailWeightedElasticNetRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, tail_weight_multiplier=1.0, lower_quantile=0.10, upper_quantile=0.90, alpha=0.1, l1_ratio=0.5, fit_intercept=True, max_iter=20000, tol=1e-4, random_state=42, selection="cyclic"):
        self.tail_weight_multiplier = tail_weight_multiplier
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.selection = selection

    def fit(self, X, y):
        y_array = np.asarray(y, dtype=float).reshape(-1)
        self.lower_threshold_ = float(np.quantile(y_array, self.lower_quantile))
        self.upper_threshold_ = float(np.quantile(y_array, self.upper_quantile))
        tail_mask = (y_array <= self.lower_threshold_) | (y_array >= self.upper_threshold_)
        sample_weight = np.where(tail_mask, float(self.tail_weight_multiplier), 1.0)
        self.regressor_ = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio, fit_intercept=self.fit_intercept, max_iter=self.max_iter, tol=self.tol, random_state=self.random_state, selection=self.selection)
        self.regressor_.fit(X, y_array, sample_weight=sample_weight)
        self.coef_ = self.regressor_.coef_
        self.intercept_ = self.regressor_.intercept_
        self.n_features_in_ = self.regressor_.n_features_in_
        if hasattr(self.regressor_, "feature_names_in_"):
            self.feature_names_in_ = self.regressor_.feature_names_in_
        return self

    def predict(self, X):
        return self.regressor_.predict(X)


TREE_FAMILY_LABELS = {
    "random_forest": "Random Forest",
    "extra_trees": "Extra Trees",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost"
}


class TailWeightedTreeRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, model_family="random_forest", tail_weight_multiplier=2.0, lower_quantile=0.10, upper_quantile=0.90, n_estimators=300, max_depth=6, minimum_leaf_control=10, max_features=1.0, learning_rate=0.05, subsample=1.0, num_leaves=31, random_state=42, n_jobs=1):
        self.model_family = model_family
        self.tail_weight_multiplier = tail_weight_multiplier
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.minimum_leaf_control = minimum_leaf_control
        self.max_features = max_features
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.num_leaves = num_leaves
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _build_regressor(self):
        if self.model_family == "random_forest":
            return RandomForestRegressor(n_estimators=self.n_estimators, max_depth=self.max_depth, min_samples_leaf=int(self.minimum_leaf_control), max_features=self.max_features, random_state=self.random_state, n_jobs=self.n_jobs)
        if self.model_family == "extra_trees":
            return ExtraTreesRegressor(n_estimators=self.n_estimators, max_depth=self.max_depth, min_samples_leaf=int(self.minimum_leaf_control), max_features=self.max_features, random_state=self.random_state, n_jobs=self.n_jobs)
        if self.model_family == "lightgbm":
            from lightgbm import LGBMRegressor
            return LGBMRegressor(objective="regression", n_estimators=self.n_estimators, learning_rate=self.learning_rate, max_depth=-1 if self.max_depth is None else int(self.max_depth), num_leaves=int(self.num_leaves), min_child_samples=int(self.minimum_leaf_control), subsample=self.subsample, subsample_freq=1, colsample_bytree=float(self.max_features), random_state=self.random_state, n_jobs=self.n_jobs, verbosity=-1)
        if self.model_family == "xgboost":
            from xgboost import XGBRegressor
            return XGBRegressor(objective="reg:squarederror", n_estimators=self.n_estimators, learning_rate=self.learning_rate, max_depth=int(self.max_depth), min_child_weight=float(self.minimum_leaf_control), subsample=self.subsample, colsample_bytree=float(self.max_features), tree_method="hist", random_state=self.random_state, n_jobs=self.n_jobs, verbosity=0)
        raise ValueError(f"Unknown model_family {self.model_family!r}. Choose from {sorted(TREE_FAMILY_LABELS)}.")

    def fit(self, X, y):
        if self.tail_weight_multiplier < 1:
            raise ValueError("tail_weight_multiplier must be at least 1.")
        y_array = np.asarray(y, dtype=float).reshape(-1)
        self.lower_threshold_ = float(np.quantile(y_array, self.lower_quantile))
        self.upper_threshold_ = float(np.quantile(y_array, self.upper_quantile))
        tail_mask = (y_array <= self.lower_threshold_) | (y_array >= self.upper_threshold_)
        sample_weight = np.where(tail_mask, float(self.tail_weight_multiplier), 1.0)
        self.regressor_ = self._build_regressor()
        self.regressor_.fit(X, y_array, sample_weight=sample_weight)
        self.feature_importances_ = np.asarray(self.regressor_.feature_importances_)
        self.n_features_in_ = self.regressor_.n_features_in_
        self.original_training_rows_ = len(y_array)
        self.tail_training_rows_ = int(tail_mask.sum())
        if hasattr(self.regressor_, "feature_names_in_"):
            self.feature_names_in_ = self.regressor_.feature_names_in_
        return self

    def predict(self, X):
        return self.regressor_.predict(X)


class TailWeightedMLPRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, tail_weight_multiplier=1, lower_quantile=0.10, upper_quantile=0.90, target_divisor=100.0, hidden_layer_sizes=(32, 16), activation="relu", solver="adam", alpha=1e-4, batch_size=64, learning_rate_init=0.001, max_iter=1500, tol=1e-4, n_iter_no_change=60, shuffle=True, random_state=42):
        self.tail_weight_multiplier = tail_weight_multiplier
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.target_divisor = target_divisor
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.tol = tol
        self.n_iter_no_change = n_iter_no_change
        self.shuffle = shuffle
        self.random_state = random_state

    def fit(self, X, y):
        if int(self.tail_weight_multiplier) != self.tail_weight_multiplier:
            raise ValueError("tail_weight_multiplier must be an integer for row replication.")
        if self.tail_weight_multiplier < 1 or self.target_divisor <= 0:
            raise ValueError("tail_weight_multiplier must be at least 1 and target_divisor must be greater than zero.")
        X_array = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
        y_array = np.asarray(y, dtype=float).reshape(-1)
        self.lower_threshold_ = float(np.quantile(y_array, self.lower_quantile))
        self.upper_threshold_ = float(np.quantile(y_array, self.upper_quantile))
        tail_mask = (y_array <= self.lower_threshold_) | (y_array >= self.upper_threshold_)
        repetitions = np.where(tail_mask, int(self.tail_weight_multiplier), 1)
        X_resampled = np.repeat(X_array, repetitions, axis=0)
        if hasattr(X, "columns"):
            X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
        y_resampled = np.repeat(y_array, repetitions) / float(self.target_divisor)
        self.regressor_ = MLPRegressor(hidden_layer_sizes=self.hidden_layer_sizes, activation=self.activation, solver=self.solver, alpha=self.alpha, batch_size=self.batch_size, learning_rate_init=self.learning_rate_init, max_iter=self.max_iter, tol=self.tol, n_iter_no_change=self.n_iter_no_change, early_stopping=False, shuffle=self.shuffle, random_state=self.random_state)
        self.regressor_.fit(X_resampled, y_resampled)
        self.n_features_in_ = self.regressor_.n_features_in_
        self.n_iter_ = self.regressor_.n_iter_
        self.converged_ = self.n_iter_ < self.max_iter
        self.initial_scaled_loss_ = float(self.regressor_.loss_curve_[0])
        self.final_scaled_loss_ = float(self.regressor_.loss_curve_[-1])
        self.original_training_rows_ = len(y_array)
        self.tail_training_rows_ = int(tail_mask.sum())
        self.resampled_training_rows_ = len(y_resampled)
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def predict(self, X):
        return np.asarray(self.regressor_.predict(X)).reshape(-1) * float(self.target_divisor)


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
