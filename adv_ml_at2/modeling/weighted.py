from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils.validation import has_fit_parameter


class WeightedRegressor(RegressorMixin, BaseEstimator):
    """
    Wrap a scikit-learn regressor with target-based sample weighting.

    Parameters
    ----------
    estimator:
        Scikit-learn compatible regression estimator.
    weighting_strategy:
        Weighting strategy used during training.
        Supported values are "none", "linear",
        "quadratic", and "quantile".
    weight_strength:
        Strength of the weighting function.
    max_weight:
        Maximum allowed sample weight.
    """
    def __init__(self, estimator, weighting_strategy="none", weight_strength=0.5, max_weight=3.0):
        self.estimator = estimator
        self.weighting_strategy = weighting_strategy
        self.weight_strength = weight_strength
        self.max_weight = max_weight

    def _build_sample_weights(self, y):
        """
        Build sample weights from the training target.

        Parameters
        ----------
        y:
            Training target values.

        Returns
        -------
        numpy.ndarray
            Sample weights aligned with the training target.
        """
        y_series = pd.Series(np.asarray(y, dtype=float).reshape(-1))
        if self.weight_strength < 0:
            raise ValueError("weight_strength must be greater than or equal to zero.")
        if self.max_weight < 1:
            raise ValueError("max_weight must be at least 1.")
        median = y_series.median()
        iqr = y_series.quantile(0.75) - y_series.quantile(0.25)
        if iqr == 0:
            distance = np.zeros(len(y_series))
        else:
            distance = ((y_series - median).abs() / iqr).to_numpy()

        match self.weighting_strategy:
            case "none":
                weights = np.ones(len(y_series))
            case "linear":
                weights = 1 + self.weight_strength * distance
            case "quadratic":
                weights = 1 + self.weight_strength * distance**2
            case "quantile":
                q10 = y_series.quantile(0.10)
                q25 = y_series.quantile(0.25)
                q75 = y_series.quantile(0.75)
                q90 = y_series.quantile(0.90)
                weights = np.ones(len(y_series))
                shoulder_mask = ((y_series > q10) & (y_series <= q25)) | ((y_series >= q75) & (y_series < q90))
                tail_mask = (y_series <= q10) | (y_series >= q90)
                weights[shoulder_mask] = 1 + self.weight_strength
                weights[tail_mask] = 1 + 2 * self.weight_strength
            case _:
                raise ValueError("weighting_strategy must be one of 'none', 'linear', 'quadratic', or 'quantile'.")

        return np.clip(weights, 1.0, self.max_weight)

    def fit(self, X, y):
        """
        Fit the wrapped estimator using target-based sample weights.

        Parameters
        ----------
        X:
            Training feature matrix.
        y:
            Training target values.

        Returns
        -------
        WeightedRegressor
            Fitted estimator wrapper.
        """
        y_array = np.asarray(y, dtype=float).reshape(-1)
        self.sample_weight_ = self._build_sample_weights(y_array)
        self.estimator_ = clone(self.estimator)
        if not has_fit_parameter(self.estimator_, "sample_weight"):
            raise ValueError(f"{type(self.estimator_).__name__} does not support sample_weight.")
        self.estimator_.fit(X, y_array, sample_weight=self.sample_weight_)
        self.n_features_in_ = self.estimator_.n_features_in_
        if hasattr(self.estimator_, "feature_names_in_"):
            self.feature_names_in_ = self.estimator_.feature_names_in_

        for attribute in ("coef_", "intercept_", "feature_importances_", "n_iter_"):
            if hasattr(self.estimator_, attribute):
                setattr(self, attribute, getattr(self.estimator_, attribute))

        self.mean_sample_weight_ = float(self.sample_weight_.mean())
        self.max_applied_weight_ = float(self.sample_weight_.max())
        return self

    def predict(self, X):
        """
        Generate predictions using the fitted estimator.

        Parameters
        ----------
        X:
            Feature matrix.

        Returns
        -------
        numpy.ndarray
            Model predictions.
        """
        return np.asarray(self.estimator_.predict(X)).reshape(-1)

def make_weighted_regressor(estimator, weighting_strategy="none", weight_strength=0.5, max_weight=3.0):
    """
    Create a weighted wrapper around a compatible regressor.

    Parameters
    ----------
    estimator:
        Scikit-learn compatible regression estimator.
    weighting_strategy:
        Weighting strategy used during training.
    weight_strength:
        Strength of the weighting function.
    max_weight:
        Maximum allowed sample weight.

    Returns
    -------
    WeightedRegressor
        Configured weighted estimator.
    """
    return WeightedRegressor(estimator=estimator, weighting_strategy=weighting_strategy, weight_strength=weight_strength, max_weight=max_weight)