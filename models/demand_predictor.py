"""
Demand Prediction Model – Random Forest Regressor.

Predicts market-wide demand given a set of company and market features.
The model is trained once on synthetic data and then queried each simulation round.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import MarketConfig, ModelConfig
from utils.data_generator import generate_demand_training_data


FEATURE_COLUMNS = [
    "price",
    "marketing_budget",
    "product_quality",
    "seasonality",
    "competitor_price",
]

TARGET_COLUMN = "market_demand"


@dataclass
class DemandModelMetrics:
    r2: float
    mae: float
    feature_importances: Dict[str, float]


class DemandPredictor:
    """
    Random Forest–based demand predictor.

    Usage
    -----
    predictor = DemandPredictor(market_cfg, model_cfg)
    predictor.train()
    demand = predictor.predict(price=100, marketing_budget=50000,
                               product_quality=0.7, seasonality=1.05,
                               competitor_price=110)
    """

    def __init__(
        self,
        market_cfg: MarketConfig = MarketConfig(),
        model_cfg: ModelConfig = ModelConfig(),
        random_state: int = 42,
    ) -> None:
        self._market_cfg = market_cfg
        self._model_cfg = model_cfg
        self._random_state = random_state

        self._model = RandomForestRegressor(
            n_estimators=model_cfg.demand_n_estimators,
            max_depth=model_cfg.demand_max_depth,
            min_samples_leaf=model_cfg.demand_min_samples_leaf,
            random_state=model_cfg.demand_random_state,
            n_jobs=-1,
        )
        self._scaler = StandardScaler()
        self._is_trained = False
        self._metrics: Optional[DemandModelMetrics] = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self) -> DemandModelMetrics:
        """
        Train the demand model on freshly generated synthetic data.
        Returns evaluation metrics on a held-out test split.
        """
        df = generate_demand_training_data(
            market_cfg=self._market_cfg,
            model_cfg=self._model_cfg,
            random_state=self._random_state,
        )

        X = df[FEATURE_COLUMNS].values
        y = df[TARGET_COLUMN].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self._random_state
        )

        X_train_scaled = self._scaler.fit_transform(X_train)
        X_test_scaled = self._scaler.transform(X_test)

        self._model.fit(X_train_scaled, y_train)
        self._is_trained = True

        y_pred = self._model.predict(X_test_scaled)
        self._metrics = DemandModelMetrics(
            r2=float(r2_score(y_test, y_pred)),
            mae=float(mean_absolute_error(y_test, y_pred)),
            feature_importances={
                col: float(imp)
                for col, imp in zip(FEATURE_COLUMNS, self._model.feature_importances_)
            },
        )
        return self._metrics

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        price: float,
        marketing_budget: float,
        product_quality: float,
        seasonality: float,
        competitor_price: float,
    ) -> float:
        """Return predicted market demand (units)."""
        self._assert_trained()
        X = np.array([[price, marketing_budget, product_quality,
                        seasonality, competitor_price]])
        X_scaled = self._scaler.transform(X)
        return float(self._model.predict(X_scaled)[0])

    def predict_batch(self, df: pd.DataFrame) -> np.ndarray:
        """Predict demand for a DataFrame with the required feature columns."""
        self._assert_trained()
        X = df[FEATURE_COLUMNS].values
        X_scaled = self._scaler.transform(X)
        return self._model.predict(X_scaled)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def metrics(self) -> Optional[DemandModelMetrics]:
        return self._metrics

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def _assert_trained(self) -> None:
        if not self._is_trained:
            raise RuntimeError(
                "DemandPredictor must be trained before calling predict(). "
                "Call .train() first."
            )
