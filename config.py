"""
Central configuration for MarketMind AI simulation.

All tuneable constants live here so nothing is hardcoded in business logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Market defaults
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MarketConfig:
    """Immutable market-level configuration."""

    market_size: float = 1_000_000.0        # total addressable units
    consumer_sensitivity: float = 0.6       # 0-1, higher → more price-driven
    price_elasticity: float = 1.5           # demand elasticity to price
    brand_loyalty: float = 0.3             # stickiness to incumbent brand
    competition_intensity: float = 0.7     # 0-1, higher → fiercer rivalry

    # Seasonality wave parameters (sinusoidal multiplier)
    seasonality_amplitude: float = 0.15
    seasonality_period: int = 12            # rounds per full cycle


# ---------------------------------------------------------------------------
# Simulation defaults
# ---------------------------------------------------------------------------

@dataclass
class SimulationConfig:
    """Mutable simulation run configuration (set from UI or code)."""

    num_companies: int = 4
    num_rounds: int = 20
    random_seed: int = 42

    # Starting ranges for company initialisation
    price_range: tuple[float, float] = (50.0, 150.0)
    marketing_range: tuple[float, float] = (10_000.0, 100_000.0)
    quality_range: tuple[float, float] = (0.4, 0.9)

    # Agent learning rate / exploration
    strategy_learning_rate: float = 0.05
    exploration_rate: float = 0.2          # ε-greedy exploration


# ---------------------------------------------------------------------------
# ML model defaults
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelConfig:
    """Hyper-parameters for ML models."""

    # Demand predictor (Random Forest)
    demand_n_estimators: int = 120
    demand_max_depth: int = 8
    demand_min_samples_leaf: int = 5
    demand_random_state: int = 42

    # Customer choice (Random Forest classifier)
    choice_n_estimators: int = 100
    choice_max_depth: int = 6
    choice_random_state: int = 42

    # Synthetic training-set size
    training_samples: int = 2_000


# ---------------------------------------------------------------------------
# Colour palette (shared across charts)
# ---------------------------------------------------------------------------

COMPANY_COLORS: List[str] = [
    "#2196F3",  # blue
    "#F44336",  # red
    "#4CAF50",  # green
    "#FF9800",  # orange
    "#9C27B0",  # purple
    "#00BCD4",  # cyan
    "#795548",  # brown
    "#607D8B",  # blue-grey
]

DEFAULT_COMPANY_NAMES: List[str] = [
    "AlphaCorp",
    "BetaTech",
    "GammaCo",
    "DeltaInc",
    "EpsilonLtd",
    "ZetaGroup",
    "EtaVentures",
    "ThetaCo",
]
