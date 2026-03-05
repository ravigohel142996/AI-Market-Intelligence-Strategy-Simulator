"""
Synthetic data generation for ML model training.

Produces realistic-looking market datasets so models can be trained
before any live simulation data is available.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import MarketConfig, ModelConfig


def _seasonality(round_idx: np.ndarray, amplitude: float, period: int) -> np.ndarray:
    """Return sinusoidal seasonality multiplier centred at 1.0."""
    return 1.0 + amplitude * np.sin(2 * np.pi * round_idx / period)


def generate_demand_training_data(
    market_cfg: MarketConfig = MarketConfig(),
    model_cfg: ModelConfig = ModelConfig(),
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic training data for the demand prediction model.

    Features
    --------
    price, marketing_budget, product_quality, seasonality, competitor_price

    Target
    ------
    market_demand  (units sold, already scaled to market_size)
    """
    rng = np.random.default_rng(random_state)
    n = model_cfg.training_samples

    price = rng.uniform(30.0, 200.0, n)
    marketing = rng.uniform(5_000.0, 200_000.0, n)
    quality = rng.uniform(0.2, 1.0, n)
    round_idx = rng.integers(0, 48, n).astype(float)
    competitor_price = rng.uniform(30.0, 200.0, n)

    seasonality = _seasonality(round_idx, market_cfg.seasonality_amplitude,
                                market_cfg.seasonality_period)

    # Demand model: price suppresses, marketing/quality boost, competitor
    # price differential adds a relative advantage.
    price_effect = (1.0 - market_cfg.consumer_sensitivity
                    * (price / 200.0) ** market_cfg.price_elasticity)
    marketing_effect = np.log1p(marketing) / np.log1p(200_000.0)
    quality_effect = quality ** 0.7
    competitor_effect = 1.0 + 0.15 * ((competitor_price - price) / 200.0)

    base_demand = (market_cfg.market_size
                   * price_effect
                   * marketing_effect
                   * quality_effect
                   * competitor_effect
                   * seasonality)

    noise = rng.normal(0.0, 0.03 * base_demand)
    market_demand = np.clip(base_demand + noise, 0, market_cfg.market_size)

    return pd.DataFrame({
        "price": price,
        "marketing_budget": marketing,
        "product_quality": quality,
        "seasonality": seasonality,
        "competitor_price": competitor_price,
        "market_demand": market_demand,
    })


def generate_choice_training_data(
    num_companies: int = 4,
    model_cfg: ModelConfig = ModelConfig(),
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic training data for the customer choice model.

    Each row is one consumer decision among *num_companies* alternatives.

    Features
    --------
    price, brand_score, marketing_strength, product_quality

    Target
    ------
    chosen_company  (integer label 0 … num_companies-1)
    """
    rng = np.random.default_rng(random_state)
    n = model_cfg.training_samples

    price = rng.uniform(30.0, 200.0, n)
    brand = rng.uniform(0.1, 1.0, n)
    marketing = rng.uniform(0.0, 1.0, n)
    quality = rng.uniform(0.2, 1.0, n)

    # Utility: lower price + higher brand/marketing/quality → higher utility
    utility = (-0.4 * (price / 200.0)
               + 0.25 * brand
               + 0.2 * marketing
               + 0.35 * quality
               + rng.normal(0.0, 0.05, n))

    # Map continuous utility to num_companies buckets
    chosen = pd.qcut(utility, q=num_companies, labels=False)

    return pd.DataFrame({
        "price": price,
        "brand_score": brand,
        "marketing_strength": marketing,
        "product_quality": quality,
        "chosen_company": chosen.astype(int),
    })
