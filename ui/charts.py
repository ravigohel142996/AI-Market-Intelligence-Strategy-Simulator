"""
Charts – thin rendering layer between analytics and Streamlit.

Each function calls the underlying chart builder and immediately renders
the resulting Plotly figure via `st.plotly_chart`.  Keeping the render
calls here prevents the dashboard from importing Plotly directly.
"""

from __future__ import annotations

import streamlit as st

from analytics.visualizations import (
    competition_index_chart,
    demand_prediction_chart,
    feature_importance_chart,
    market_share_evolution,
    market_share_pie,
    profit_trend,
    strategy_radar,
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
