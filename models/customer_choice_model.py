"""
Customer Choice Model – Random Forest Classifier.

Estimates the probability that a consumer chooses each company given the
competing offers visible in the market at any point in time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from config import ModelConfig
from utils.data_generator import generate_choice_training_data


FEATURE_COLUMNS = [
    "price",
    "brand_score",
    "marketing_strength",
    "product_quality",
]

TARGET_COLUMN = "chosen_company"


@dataclass
class ChoiceModelMetrics:
    accuracy: float
    feature_importances: Dict[str, float]
    num_classes: int


class CustomerChoiceModel:
    """
    Random Forest–based customer choice classifier.

    Given the attributes of every company in the market, the model
    returns market-share probabilities for each company.

    Usage
    -----
    model = CustomerChoiceModel(num_companies=4)
    model.train()
    probs = model.predict_shares(companies_df)   # one row per company
    """

    def __init__(
        self,
        num_companies: int = 4,
        model_cfg: ModelConfig = ModelConfig(),
        random_state: int = 42,
    ) -> None:
        self._num_companies = num_companies
        self._model_cfg = model_cfg
        self._random_state = random_state

        self._model = RandomForestClassifier(
            n_estimators=model_cfg.choice_n_estimators,
            max_depth=model_cfg.choice_max_depth,
            random_state=model_cfg.choice_random_state,
            n_jobs=-1,
        )
        self._scaler = StandardScaler()
        self._label_enc = LabelEncoder()
        self._is_trained = False
        self._metrics: Optional[ChoiceModelMetrics] = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self) -> ChoiceModelMetrics:
        """Train classifier on synthetic choice data."""
        df = generate_choice_training_data(
            num_companies=self._num_companies,
            model_cfg=self._model_cfg,
            random_state=self._random_state,
        )

        X = df[FEATURE_COLUMNS].values
        y = self._label_enc.fit_transform(df[TARGET_COLUMN].values)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self._random_state
        )

        X_train_scaled = self._scaler.fit_transform(X_train)
        X_test_scaled = self._scaler.transform(X_test)

        self._model.fit(X_train_scaled, y_train)
        self._is_trained = True

        y_pred = self._model.predict(X_test_scaled)
        self._metrics = ChoiceModelMetrics(
            accuracy=float(accuracy_score(y_test, y_pred)),
            feature_importances={
                col: float(imp)
                for col, imp in zip(FEATURE_COLUMNS, self._model.feature_importances_)
            },
            num_classes=len(self._label_enc.classes_),
        )
        return self._metrics

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_shares(self, companies_df: pd.DataFrame) -> List[float]:
        """
        Given a DataFrame with one row per company (columns = FEATURE_COLUMNS),
        return normalised market-share probabilities summing to 1.0.

        The method averages per-class probabilities across all rows (each row
        represents one simulated consumer), producing aggregate share estimates.
        """
        self._assert_trained()

        X = companies_df[FEATURE_COLUMNS].values
        X_scaled = self._scaler.transform(X)

        # predict_proba: shape (n_companies, n_classes)
        proba_matrix = self._model.predict_proba(X_scaled)

        # Each row already sums to 1 — take the diagonal as each company's
        # probability of being chosen *given its own feature vector*.
        # The diagonal gives P(class=i | features of company i).
        n_companies = len(companies_df)
        n_classes = proba_matrix.shape[1]

        raw_shares: List[float] = []
        for i in range(n_companies):
            # The diagonal gives P(class=i | features of company i):
            # each row i in proba_matrix is the classifier's probability
            # distribution over all classes given company i's feature vector.
            # Taking index i (clamped to n_classes-1) treats each company as
            # its own "class", producing a relative attractiveness score that
            # we then normalise into market shares.
            class_idx = min(i, n_classes - 1)
            raw_shares.append(float(proba_matrix[i, class_idx]))

        # Normalise so shares sum to 1
        total = sum(raw_shares)
        if total <= 0:
            return [1.0 / n_companies] * n_companies
        return [s / total for s in raw_shares]

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def metrics(self) -> Optional[ChoiceModelMetrics]:
        return self._metrics

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def _assert_trained(self) -> None:
        if not self._is_trained:
            raise RuntimeError(
                "CustomerChoiceModel must be trained before calling predict_shares(). "
                "Call .train() first."
            )
