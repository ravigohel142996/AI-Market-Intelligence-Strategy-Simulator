"""
Scenario Analyzer – competitive intelligence and strategy effectiveness.

Provides insights into how the simulation evolved by analysing price
positioning, marketing return-on-investment, and demand trends across
all simulation rounds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from core.simulation_engine import SimulationResult
from utils.helpers import pct_change


@dataclass
class MarketingROI:
    """Marketing return-on-investment summary for one company."""

    name: str
    avg_marketing_spend: float
    avg_market_share: float
    roi_score: float          # avg_market_share / normalised_marketing_spend


@dataclass
class PricePositioning:
    """Price–quality positioning for one company in the final round."""

    name: str
    price: float
    quality: float
    market_share: float
    value_score: float        # quality / price × 100 (higher = better value)
    positioning: str          # 'Premium', 'Value', 'Mid-range'


class ScenarioAnalyzer:
    """
    Analyses competitive dynamics and strategy effectiveness from a
    SimulationResult.  All methods are pure – no side effects.
    """

    def __init__(self, result: SimulationResult) -> None:
        self._result = result

    # ------------------------------------------------------------------
    # Marketing ROI
    # ------------------------------------------------------------------

    def marketing_roi(self) -> List[MarketingROI]:
        """Compute marketing ROI for each company (final-round spend, avg share)."""
        shares_df = self._result.shares_df()
        final = self._result.final_states()

        max_marketing = max((c.marketing_budget for c in final), default=1.0)
        roi_list: List[MarketingROI] = []

        for cs in final:
            if cs.name not in shares_df.columns:
                continue
            avg_share = float(shares_df[cs.name].mean())
            # Guard against companies with zero marketing spend
            if cs.marketing_budget <= 0:
                roi = 0.0
            else:
                marketing_fraction = cs.marketing_budget / max_marketing
                roi = avg_share / marketing_fraction
            roi_list.append(MarketingROI(
                name=cs.name,
                avg_marketing_spend=cs.marketing_budget,
                avg_market_share=avg_share,
                roi_score=roi,
            ))

        return sorted(roi_list, key=lambda r: r.roi_score, reverse=True)

    # ------------------------------------------------------------------
    # Price–quality positioning
    # ------------------------------------------------------------------

    def price_positioning(self) -> List[PricePositioning]:
        """Analyse price–quality positioning in the final round."""
        final = self._result.final_states()
        all_prices = [c.price for c in final]
        price_p33 = float(np.percentile(all_prices, 33))
        price_p66 = float(np.percentile(all_prices, 66))

        positions: List[PricePositioning] = []
        for cs in final:
            value_score = cs.product_quality / max(cs.price, 1.0) * 100
            if cs.price >= price_p66:
                positioning = "Premium"
            elif cs.price <= price_p33:
                positioning = "Value"
            else:
                positioning = "Mid-range"

            positions.append(PricePositioning(
                name=cs.name,
                price=cs.price,
                quality=cs.product_quality,
                market_share=cs.market_share,
                value_score=round(value_score, 4),
                positioning=positioning,
            ))

        return sorted(positions, key=lambda p: p.value_score, reverse=True)

    # ------------------------------------------------------------------
    # Combined summary table
    # ------------------------------------------------------------------

    def competitive_summary_table(self) -> pd.DataFrame:
        """Combined summary: price–quality positioning + marketing ROI."""
        roi_map = {r.name: r for r in self.marketing_roi()}
        rows = []
        for pp in self.price_positioning():
            roi = roi_map.get(pp.name)
            rows.append({
                "Company": pp.name,
                "Price ($)": round(pp.price, 2),
                "Quality": round(pp.quality, 3),
                "Value Score": round(pp.value_score, 4),
                "Positioning": pp.positioning,
                "Marketing ROI": round(roi.roi_score, 4) if roi else 0.0,
                "Avg Share": f"{roi.avg_market_share * 100:.1f}%" if roi else "N/A",
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Demand trend summary
    # ------------------------------------------------------------------

    def demand_trend_summary(self) -> Dict[str, float]:
        """Summarise demand trend across the full simulation."""
        demands = [rr.total_demand for rr in self._result.rounds]
        if len(demands) < 2:
            return {"first_half_avg": demands[0] if demands else 0.0,
                    "second_half_avg": demands[-1] if demands else 0.0,
                    "trend_pct": 0.0,
                    "peak_demand": float(max(demands)) if demands else 0.0,
                    "trough_demand": float(min(demands)) if demands else 0.0}
        mid = len(demands) // 2
        first_half = demands[:mid]
        second_half = demands[mid:]
        return {
            "first_half_avg": float(np.mean(first_half)),
            "second_half_avg": float(np.mean(second_half)),
            "trend_pct": pct_change(float(np.mean(first_half)), float(np.mean(second_half))),
            "peak_demand": float(max(demands)),
            "trough_demand": float(min(demands)),
        }
