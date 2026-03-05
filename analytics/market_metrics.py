"""
Market Metrics – aggregation and derived KPI computation.

Transforms raw SimulationResult data into summary statistics used by
the dashboard and export functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from core.simulation_engine import SimulationResult
from core.market_environment import CompanyState
from utils.helpers import fmt_currency, fmt_pct, pct_change


@dataclass
class KPISnapshot:
    """High-level KPIs derived from the final simulation round."""

    total_market_value: float
    market_leader_name: str
    market_leader_share: float
    total_demand: float
    competition_index: float
    avg_price: float
    avg_quality: float


@dataclass
class CompanySummary:
    """Per-company summary statistics across all rounds."""

    name: str
    final_share: float
    final_profit: float
    peak_profit: float
    peak_share: float
    avg_share: float
    avg_profit: float
    share_trend: float       # % change first → last round
    profit_trend: float
    last_action: Optional[str]


class MarketMetrics:
    """
    Stateless analytics layer that operates on a SimulationResult.
    All methods are pure functions – no side effects.
    """

    def __init__(self, result: SimulationResult) -> None:
        self._result = result

    # ------------------------------------------------------------------
    # Top-level KPIs
    # ------------------------------------------------------------------

    def kpi_snapshot(self) -> KPISnapshot:
        """Extract headline KPIs from the final round."""
        last = self._result.final_states()
        if not last:
            return KPISnapshot(0, "N/A", 0, 0, 0, 0, 0)

        leader = max(last, key=lambda c: c.market_share)
        last_round = self._result.rounds[-1]

        return KPISnapshot(
            total_market_value=sum(
                c.price * c.market_share * last_round.total_demand for c in last
            ),
            market_leader_name=leader.name,
            market_leader_share=leader.market_share,
            total_demand=last_round.total_demand,
            competition_index=last_round.competition_index,
            avg_price=float(np.mean([c.price for c in last])),
            avg_quality=float(np.mean([c.product_quality for c in last])),
        )

    # ------------------------------------------------------------------
    # Per-company summaries
    # ------------------------------------------------------------------

    def company_summaries(self) -> List[CompanySummary]:
        """Build a CompanySummary for every company."""
        shares_df = self._result.shares_df()
        profits_df = self._result.profits_df()
        names = self._result.company_names

        summaries: List[CompanySummary] = []
        for name in names:
            if name not in shares_df.columns:
                continue

            shares = shares_df[name].values
            profits = profits_df[name].values

            summaries.append(CompanySummary(
                name=name,
                final_share=float(shares[-1]),
                final_profit=float(profits[-1]),
                peak_profit=float(profits.max()),
                peak_share=float(shares.max()),
                avg_share=float(shares.mean()),
                avg_profit=float(profits.mean()),
                share_trend=pct_change(shares[0], shares[-1]),
                profit_trend=pct_change(profits[0], profits[-1]),
                last_action=self._last_action_for(name),
            ))

        return sorted(summaries, key=lambda s: s.final_share, reverse=True)

    def company_table(self) -> pd.DataFrame:
        """
        Returns a formatted DataFrame suitable for display in the dashboard.
        Columns: Company, Price, Marketing, Quality, Market Share, Profit
        """
        final = self._result.final_states()
        rows = []
        for cs in final:
            rows.append({
                "Company": cs.name,
                "Price ($)": round(cs.price, 2),
                "Marketing ($)": round(cs.marketing_budget, 0),
                "Quality": round(cs.product_quality, 3),
                "Market Share": fmt_pct(cs.market_share),
                "Profit ($)": round(cs.profit, 0),
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Demand metrics
    # ------------------------------------------------------------------

    def demand_statistics(self) -> Dict[str, float]:
        """Summary statistics about predicted demand over the simulation."""
        demands = [rr.total_demand for rr in self._result.rounds]
        return {
            "mean": float(np.mean(demands)),
            "std": float(np.std(demands)),
            "min": float(np.min(demands)),
            "max": float(np.max(demands)),
            "peak_round": int(np.argmax(demands)) + 1,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _last_action_for(self, name: str) -> Optional[str]:
        """Retrieve the last recorded action name for a company."""
        for rr in reversed(self._result.rounds):
            for cs in rr.company_states:
                if cs.name == name:
                    return None  # CompanyState doesn't carry action
        return None
