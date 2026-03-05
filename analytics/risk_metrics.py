"""
Risk Metrics – volatility and risk analysis for company performance.

Computes per-company risk indicators from the simulation history,
including return volatility, Sharpe-style ratios, max drawdown, and
profit correlation across companies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from core.simulation_engine import SimulationResult


@dataclass
class CompanyRisk:
    """Risk profile for a single company."""

    name: str
    share_volatility: float      # std dev of market share across rounds
    profit_volatility: float     # std dev of profit across rounds (absolute)
    sharpe_ratio: float          # mean return / std dev (higher = better risk-adjusted)
    max_drawdown: float          # largest peak-to-trough drop in market share
    stability_score: float       # 0-1, higher = more stable performer


# Scales share std-dev (typically 0–0.1 for 10 companies, 0–0.5 in extreme cases)
# to a 0–1 stability score.  A volatility of 0.1 (10 pp swing) → stability ≈ 0.
_STABILITY_VOLATILITY_SCALE = 10.0


class RiskMetrics:
    """
    Computes risk and volatility metrics from a completed simulation.

    All methods are pure functions – no side effects.
    """

    def __init__(self, result: SimulationResult) -> None:
        self._result = result

    # ------------------------------------------------------------------
    # Per-company risk profiles
    # ------------------------------------------------------------------

    def company_risk_profiles(self) -> List[CompanyRisk]:
        """Return a risk profile for each company, sorted by Sharpe ratio (desc)."""
        shares_df = self._result.shares_df()
        profits_df = self._result.profits_df()

        profiles: List[CompanyRisk] = []
        for name in self._result.company_names:
            if name not in shares_df.columns:
                continue

            shares = shares_df[name].values.astype(float)
            profits = profits_df[name].values.astype(float)

            share_vol = float(np.std(shares))
            profit_vol = float(np.std(profits))

            mean_profit = float(np.mean(profits))
            sharpe = mean_profit / profit_vol if profit_vol > 0 else 0.0

            max_dd = self._max_drawdown(shares)

            # Stability: low share volatility → high stability
            stability = float(np.clip(1.0 - share_vol * _STABILITY_VOLATILITY_SCALE, 0.0, 1.0))

            profiles.append(CompanyRisk(
                name=name,
                share_volatility=share_vol,
                profit_volatility=profit_vol,
                sharpe_ratio=sharpe,
                max_drawdown=max_dd,
                stability_score=stability,
            ))

        return sorted(profiles, key=lambda r: r.sharpe_ratio, reverse=True)

    def risk_table(self) -> pd.DataFrame:
        """DataFrame suitable for display in the dashboard."""
        profiles = self.company_risk_profiles()
        rows = []
        for p in profiles:
            rows.append({
                "Company": p.name,
                "Share Volatility (%)": round(p.share_volatility * 100, 2),
                "Profit Volatility ($K)": round(p.profit_volatility / 1_000, 1),
                "Sharpe Ratio": round(p.sharpe_ratio, 3),
                "Max Drawdown (%)": round(p.max_drawdown * 100, 2),
                "Stability Score": round(p.stability_score, 3),
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Correlation analysis
    # ------------------------------------------------------------------

    def profit_correlation_matrix(self) -> pd.DataFrame:
        """Return the Pearson correlation matrix of profits across companies."""
        profits_df = self._result.profits_df()
        company_cols = [
            c for c in self._result.company_names if c in profits_df.columns
        ]
        return profits_df[company_cols].corr()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _max_drawdown(series: np.ndarray) -> float:
        """Compute the maximum peak-to-trough drawdown in a series."""
        if len(series) == 0:
            return 0.0
        peak = series[0]
        max_dd = 0.0
        for val in series:
            if val > peak:
                peak = val
            dd = (peak - val) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, dd)
        return max_dd
