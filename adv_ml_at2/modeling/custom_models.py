"""Custom model wrappers and factories used across experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.utils.validation import check_is_fitted, has_fit_parameter


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
    return WeightedRegressor(
        estimator=estimator,
        weighting_strategy=weighting_strategy,
        weight_strength=weight_strength,
        max_weight=max_weight
    )


class SMOTEClassifier(ClassifierMixin, BaseEstimator):
    """
    Wrap a scikit-learn classifier with proportion-aware SMOTE oversampling.

    SMOTE is applied only during ``fit``. Validation, test, and inference
    observations are never resampled.

    Parameters
    ----------
    estimator:
        Scikit-learn compatible classification estimator.
    sampling_multiplier:
        Maximum oversampling strength. The actual increase for each minority
        class is reduced according to how close that class already is to the
        majority-class size. A value of 1.0 applies no oversampling.
    k_neighbors:
        Number of nearest neighbours used by SMOTE. If a class selected for
        oversampling has too few observations, the effective value is reduced
        automatically to remain valid.
    random_state:
        Random state passed to SMOTE.
    """
    def __init__(
        self,
        estimator,
        sampling_multiplier=1.0,
        k_neighbors=3,
        random_state=42
    ):
        self.estimator = estimator
        self.sampling_multiplier = sampling_multiplier
        self.k_neighbors = k_neighbors
        self.random_state = random_state

    def _build_sampling_strategy(self, y):
        """
        Build a proportion-aware SMOTE sampling strategy.

        The requested multiplier is scaled by class rarity:

            rarity = 1 - current_count / majority_count

            extra_samples =
                current_count
                * (sampling_multiplier - 1)
                * rarity

            target_count = current_count + extra_samples

        Very rare classes therefore receive close to the full requested
        multiplier, while classes already close to the majority receive
        a much smaller increase.

        Parameters
        ----------
        y:
            Training class labels.

        Returns
        -------
        dict | None
            Target class counts accepted by ``imblearn.over_sampling.SMOTE``.
        """
        y_series = pd.Series(np.asarray(y).reshape(-1))
        class_counts = y_series.value_counts()

        if len(class_counts) < 2:
            raise ValueError("SMOTEClassifier requires at least two classes.")
        if self.sampling_multiplier < 1:
            raise ValueError("sampling_multiplier must be at least 1.")

        majority_count = int(class_counts.max())
        strategy = {}

        for class_label, current_count in class_counts.items():
            current_count = int(current_count)

            if current_count == majority_count:
                continue

            rarity = 1 - current_count / majority_count
            extra_samples = current_count * (self.sampling_multiplier - 1) * rarity
            target_count = min(
                int(np.ceil(current_count + extra_samples)),
                majority_count
            )

            if target_count > current_count:
                strategy[class_label] = target_count

        return strategy or None

    def _effective_k_neighbors(self, y, sampling_strategy):
        """
        Determine a valid neighbour count for the classes being oversampled.
        """
        if sampling_strategy is None:
            return self.k_neighbors
        if self.k_neighbors < 1:
            raise ValueError("k_neighbors must be at least 1.")
        y_series = pd.Series(np.asarray(y).reshape(-1))
        class_counts = y_series.value_counts()
        classes_to_resample = list(sampling_strategy)

        minimum_class_count = int(class_counts.loc[classes_to_resample].min())
        if minimum_class_count < 2:
            raise ValueError("SMOTE requires at least 2 observations in every class selected for oversampling.")
        return min(self.k_neighbors, minimum_class_count - 1)

    def fit(self, X, y):
        """
        Resample minority training classes with SMOTE and fit the classifier.

        Parameters
        ----------
        X:
            Training feature matrix.
        y:
            Training class labels.

        Returns
        -------
        SMOTEClassifier
            Fitted classifier wrapper.
        """
        y_array = np.asarray(y).reshape(-1)
        self.estimator_ = clone(self.estimator)

        self.class_counts_before_ = pd.Series(y_array).value_counts().sort_index().to_dict()
        self.sampling_strategy_ = self._build_sampling_strategy(y_array)

        if self.sampling_strategy_ is None:
            X_resampled = X
            y_resampled = y_array
            self.effective_k_neighbors_ = None
        else:
            self.effective_k_neighbors_ = self._effective_k_neighbors(y_array, self.sampling_strategy_)

            self.smote_ = SMOTE(
                sampling_strategy=self.sampling_strategy_,
                k_neighbors=self.effective_k_neighbors_,
                random_state=self.random_state
            )
            X_resampled, y_resampled = self.smote_.fit_resample(X, y_array)

        self.class_counts_after_ = pd.Series(y_resampled).value_counts().sort_index().to_dict()
        self.estimator_.fit(X_resampled, y_resampled)
        self.classes_ = self.estimator_.classes_
        if hasattr(self.estimator_, "n_features_in_"):
            self.n_features_in_ = self.estimator_.n_features_in_
        if hasattr(self.estimator_, "feature_names_in_"):
            self.feature_names_in_ = self.estimator_.feature_names_in_

        for attribute in ("coef_", "intercept_", "feature_importances_", "n_iter_"):
            if hasattr(self.estimator_, attribute):
                setattr(self, attribute, getattr(self.estimator_, attribute))
        return self

    def predict(self, X):
        """
        Generate class predictions using the fitted classifier.
        """
        check_is_fitted(self, "estimator_")
        return np.asarray(self.estimator_.predict(X)).reshape(-1)

    def predict_proba(self, X):
        """
        Generate class probabilities when supported by the estimator.
        """
        check_is_fitted(self, "estimator_")
        if not hasattr(self.estimator_, "predict_proba"):
            raise AttributeError(f"{type(self.estimator_).__name__} does not support predict_proba.")
        return self.estimator_.predict_proba(X)

    def decision_function(self, X):
        """
        Return decision scores when supported by the estimator.
        """
        check_is_fitted(self, "estimator_")
        if not hasattr(self.estimator_, "decision_function"):
            raise AttributeError(f"{type(self.estimator_).__name__} does not support decision_function.")
        return self.estimator_.decision_function(X)

def make_smote_classifier(
    estimator,
    sampling_multiplier=1.0,
    k_neighbors=3,
    random_state=42
):
    """
    Create a classifier wrapper with proportion-aware SMOTE oversampling.

    Parameters
    ----------
    estimator:
        Scikit-learn compatible classification estimator.
    sampling_multiplier:
        Maximum oversampling strength. The effective increase for each
        minority class is scaled by its rarity relative to the majority class.
    k_neighbors:
        Number of nearest neighbours used by SMOTE.
    random_state:
        Random state passed to SMOTE.

    Returns
    -------
    SMOTEClassifier
        Configured classifier wrapper.
    """
    return SMOTEClassifier(
        estimator=estimator,
        sampling_multiplier=sampling_multiplier,
        k_neighbors=k_neighbors,
        random_state=random_state
    )