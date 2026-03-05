"""
Market Dynamics – updates market share and profit after every simulation round.

This module is the economic heart of the simulation.  Given the ML model
outputs (aggregate demand, customer choice probabilities) and all companies'
current decisions, it computes the resulting market shares and profits.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config import MarketConfig
from core.market_environment import CompanyState, MarketEnvironment
from models.customer_choice_model import CustomerChoiceModel
from utils.helpers import clamp, normalise


# Cost model constants (fraction of marketing spend that is marginal cost)
_MARKETING_COST_FRACTION = 0.80
_QUALITY_UNIT_COST = 50_000.0   # additional cost per unit of quality above 0.5


class MarketDynamics:
    """
    Stateless calculator that applies economic rules to produce new
    market shares and profits for every company in a round.
    """

    def __init__(
        self,
        market_cfg: MarketConfig,
        choice_model: CustomerChoiceModel,
    ) -> None:
        self._market_cfg = market_cfg
        self._choice_model = choice_model

    # ------------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------------

    def compute_round_outcomes(
        self,
        companies: List[CompanyState],
        total_demand: float,
    ) -> List[Tuple[float, float]]:
        """
        Compute (new_market_share, new_profit) for each company.

        Parameters
        ----------
        companies     : list of company states *after* agents have decided
        total_demand  : predicted market demand for this round

        Returns
        -------
        List of (market_share, profit) tuples in the same order as *companies*.
        """
        ml_shares = self._ml_market_shares(companies)
        gravity_shares = self._gravity_shares(companies)

        # Blend ML choice model with attraction-gravity model
        cfg = self._market_cfg
        blend = cfg.brand_loyalty  # higher loyalty → more weight on gravity/brand
        blended = [
            (1 - blend) * ml_shares[i] + blend * gravity_shares[i]
            for i in range(len(companies))
        ]
        blended = normalise(blended)

        results: List[Tuple[float, float]] = []
        for i, company in enumerate(companies):
            share = blended[i]
            profit = self._compute_profit(company, share, total_demand)
            results.append((share, profit))

        return results

    # ------------------------------------------------------------------
    # Share models
    # ------------------------------------------------------------------

    def _ml_market_shares(self, companies: List[CompanyState]) -> List[float]:
        """Use the customer-choice ML model to estimate shares."""
        rows = [
            {
                "price": c.price,
                "brand_score": c.brand_score,
                "marketing_strength": c.marketing_budget / 200_000.0,
                "product_quality": c.product_quality,
            }
            for c in companies
        ]
        df = pd.DataFrame(rows)
        return self._choice_model.predict_shares(df)

    def _gravity_shares(self, companies: List[CompanyState]) -> List[float]:
        """
        Attraction model: utility ∝ quality × marketing / price^elasticity.
        Incorporates brand loyalty and competition intensity.
        """
        cfg = self._market_cfg
        utilities: List[float] = []
        for c in companies:
            u = (
                (c.product_quality ** 0.6)
                * ((c.marketing_budget / 200_000.0) ** 0.3)
                * (c.brand_score ** 0.2)
                / max(c.price, 1.0) ** cfg.price_elasticity
            )
            utilities.append(max(u, 0.0))
        return normalise(utilities)

    # ------------------------------------------------------------------
    # Profit model
    # ------------------------------------------------------------------

    def _compute_profit(
        self,
        company: CompanyState,
        share: float,
        total_demand: float,
    ) -> float:
        """
        Revenue – costs.

        Revenue  = price × (share × total_demand)
        Costs    = variable_cost + marketing_cost + quality_cost
        """
        units_sold = share * total_demand
        revenue = company.price * units_sold

        # Variable production cost: 35 % of price per unit
        variable_cost = 0.35 * company.price * units_sold

        # Marketing cost (actual spend)
        marketing_cost = company.marketing_budget * _MARKETING_COST_FRACTION

        # Quality investment cost (premium above base quality)
        quality_premium = max(company.product_quality - 0.5, 0.0)
        quality_cost = quality_premium * _QUALITY_UNIT_COST

        profit = revenue - variable_cost - marketing_cost - quality_cost
        return float(profit)
