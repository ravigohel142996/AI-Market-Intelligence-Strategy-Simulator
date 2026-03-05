"""
Market Environment – immutable state snapshot for a single simulation round.

The MarketEnvironment captures everything about the external market at a
given point in time: seasonality, aggregate demand, and the competitive
landscape produced by all companies' decisions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

from config import MarketConfig


@dataclass
class CompanyState:
    """Lightweight view of one company's current market position."""

    company_id: int
    name: str
    price: float
    marketing_budget: float
    product_quality: float   # 0.0 – 1.0
    brand_score: float       # 0.0 – 1.0, accumulated brand equity
    market_share: float      # 0.0 – 1.0
    profit: float


@dataclass
class MarketEnvironment:
    """
    Encapsulates all market-level state for a simulation round.

    The environment is read-only from the perspective of agents – they
    observe it and then submit decisions that the simulation engine applies
    to produce the next environment.
    """

    round_number: int
    market_cfg: MarketConfig
    companies: List[CompanyState]

    # Derived / updated each round
    total_demand: float = 0.0
    seasonality_factor: float = 1.0
    competition_index: float = 0.0   # 0-1 measure of market concentration

    def compute_seasonality(self) -> float:
        """Sinusoidal seasonality multiplier for the current round."""
        amplitude = self.market_cfg.seasonality_amplitude
        period = self.market_cfg.seasonality_period
        factor = 1.0 + amplitude * math.sin(2 * math.pi * self.round_number / period)
        return factor

    def compute_competition_index(self) -> float:
        """
        Herfindahl-Hirschman Index (HHI) normalised to [0, 1].

        HHI = Σ(market_share_i)²
        Max HHI (monopoly) = 1.0, min approaches 1/n for perfect competition.
        """
        if not self.companies:
            return 0.0
        hhi = sum(c.market_share ** 2 for c in self.companies)
        return float(hhi)

    def average_competitor_price(self, exclude_company_id: int) -> float:
        """Return mean price of all competitors excluding *exclude_company_id*."""
        prices = [
            c.price for c in self.companies if c.company_id != exclude_company_id
        ]
        return float(sum(prices) / len(prices)) if prices else 0.0

    def market_leader(self) -> CompanyState:
        """Return the company with the highest market share."""
        return max(self.companies, key=lambda c: c.market_share)

    def total_market_value(self) -> float:
        """Sum of (price × market_share × total_demand) across all companies."""
        return sum(
            c.price * c.market_share * self.total_demand for c in self.companies
        )

    def refresh_derived_metrics(self) -> None:
        """Recompute seasonality_factor and competition_index in place."""
        self.seasonality_factor = self.compute_seasonality()
        self.competition_index = self.compute_competition_index()
