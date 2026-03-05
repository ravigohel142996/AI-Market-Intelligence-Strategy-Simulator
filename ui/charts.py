"""
Charts – thin rendering layer between analytics and Streamlit.

Each function calls the underlying chart builder and immediately renders
the resulting Plotly figure via `st.plotly_chart`.  Keeping the render
calls here prevents the dashboard from importing Plotly directly.
"""

from __future__ import annotations

import streamlit as st

from analytics.risk_metrics import RiskMetrics
from analytics.visualizations import (
    animated_market_share_race,
    competition_index_chart,
    correlation_heatmap,
    demand_prediction_chart,
    feature_importance_chart,
    market_share_evolution,
    market_share_pie,
    marketing_roi_scatter,
    price_quality_bubble,
    profit_trend,
    strategy_radar,
    volatility_chart,
)
from core.simulation_engine import SimulationResult


def _chart_kwargs(key: str) -> dict:
    """Shared kwargs for st.plotly_chart calls."""
    return dict(use_container_width=True, key=key)


def render_market_share_pie(result: SimulationResult) -> None:
    fig = market_share_pie(result)
    st.plotly_chart(fig, **_chart_kwargs("pie_share"))


def render_market_share_evolution(result: SimulationResult) -> None:
    fig = market_share_evolution(result)
    st.plotly_chart(fig, **_chart_kwargs("share_evolution"))


def render_profit_trend(result: SimulationResult) -> None:
    fig = profit_trend(result)
    st.plotly_chart(fig, **_chart_kwargs("profit_trend"))


def render_demand_prediction(result: SimulationResult) -> None:
    fig = demand_prediction_chart(result)
    st.plotly_chart(fig, **_chart_kwargs("demand_prediction"))


def render_strategy_radar(result: SimulationResult) -> None:
    fig = strategy_radar(result)
    st.plotly_chart(fig, **_chart_kwargs("strategy_radar"))


def render_demand_feature_importance(result: SimulationResult) -> None:
    if result.demand_metrics:
        fig = feature_importance_chart(
            result.demand_metrics.feature_importances,
            title="Demand Model – Feature Importance",
        )
        st.plotly_chart(fig, **_chart_kwargs("demand_fi"))


def render_choice_feature_importance(result: SimulationResult) -> None:
    if result.choice_metrics:
        fig = feature_importance_chart(
            result.choice_metrics.feature_importances,
            title="Choice Model – Feature Importance",
        )
        st.plotly_chart(fig, **_chart_kwargs("choice_fi"))


def render_competition_index(result: SimulationResult) -> None:
    fig = competition_index_chart(result)
    st.plotly_chart(fig, **_chart_kwargs("competition_index"))


# ---------------------------------------------------------------------------
# New animated and advanced charts
# ---------------------------------------------------------------------------

def render_animated_market_share_race(result: SimulationResult) -> None:
    """Animated bar chart race – shows market-share evolution by round."""
    fig = animated_market_share_race(result)
    st.plotly_chart(fig, **_chart_kwargs("animated_race"))


def render_price_quality_bubble(result: SimulationResult) -> None:
    """Bubble chart: price vs quality, bubble size = market share."""
    fig = price_quality_bubble(result)
    st.plotly_chart(fig, **_chart_kwargs("price_quality_bubble"))


def render_volatility_chart(result: SimulationResult) -> None:
    """Horizontal bar chart of share volatility per company."""
    risk = RiskMetrics(result)
    profiles = risk.company_risk_profiles()
    fig = volatility_chart(profiles)
    st.plotly_chart(fig, **_chart_kwargs("volatility_chart"))


def render_correlation_heatmap(result: SimulationResult) -> None:
    """Heatmap of profit correlations between companies."""
    risk = RiskMetrics(result)
    corr = risk.profit_correlation_matrix()
    fig = correlation_heatmap(corr)
    st.plotly_chart(fig, **_chart_kwargs("corr_heatmap"))


def render_marketing_roi_scatter(result: SimulationResult) -> None:
    """Scatter: marketing spend vs average market share."""
    fig = marketing_roi_scatter(result)
    st.plotly_chart(fig, **_chart_kwargs("marketing_roi"))

