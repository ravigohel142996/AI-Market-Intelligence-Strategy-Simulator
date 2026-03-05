"""
Visualisations – Plotly chart builders for the MarketMind AI dashboard.

Every function accepts domain objects (SimulationResult, DataFrames) and
returns a fully configured `plotly.graph_objects.Figure` ready for rendering.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import COMPANY_COLORS
from core.simulation_engine import SimulationResult


# ---------------------------------------------------------------------------
# Shared theme
# ---------------------------------------------------------------------------

_LAYOUT_DEFAULTS = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", size=12, color="#E0E0E0"),
    margin=dict(l=40, r=20, t=50, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
)


def _color_map(names: List[str], colors: List[str]) -> dict:
    colors_ext = (colors * ((len(names) // len(colors)) + 1))[:len(names)]
    return dict(zip(names, colors_ext))


# ---------------------------------------------------------------------------
# 1. Market Share – Pie
# ---------------------------------------------------------------------------

def market_share_pie(result: SimulationResult) -> go.Figure:
    """Pie chart of final-round market shares."""
    final = result.final_states()
    names = [cs.name for cs in final]
    values = [cs.market_share for cs in final]
    colors = result.company_colors

    fig = go.Figure(go.Pie(
        labels=names,
        values=values,
        marker=dict(colors=colors, line=dict(color="#1E1E2E", width=2)),
        textinfo="label+percent",
        hole=0.4,
        hovertemplate="%{label}<br>Share: %{percent}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Final Market Share Distribution", x=0.5),
        **_LAYOUT_DEFAULTS,
    )
    return fig


# ---------------------------------------------------------------------------
# 2. Market Share Evolution – Line
# ---------------------------------------------------------------------------

def market_share_evolution(result: SimulationResult) -> go.Figure:
    """Line chart showing market share per company over all rounds."""
    df = result.shares_df()
    color_map = _color_map(result.company_names, result.company_colors)

    fig = go.Figure()
    for name in result.company_names:
        if name in df.columns:
            fig.add_trace(go.Scatter(
                x=df["round"],
                y=df[name] * 100,
                name=name,
                mode="lines+markers",
                line=dict(color=color_map[name], width=2),
                marker=dict(size=5),
                hovertemplate=f"<b>{name}</b><br>Round %{{x}}<br>Share: %{{y:.1f}}%<extra></extra>",
            ))

    fig.update_layout(
        title=dict(text="Market Share Evolution", x=0.5),
        xaxis_title="Round",
        yaxis_title="Market Share (%)",
        **_LAYOUT_DEFAULTS,
    )
    return fig


# ---------------------------------------------------------------------------
# 3. Profit Trend – Line
# ---------------------------------------------------------------------------

def profit_trend(result: SimulationResult) -> go.Figure:
    """Line chart of profit per company across rounds."""
    df = result.profits_df()
    color_map = _color_map(result.company_names, result.company_colors)

    fig = go.Figure()
    for name in result.company_names:
        if name in df.columns:
            fig.add_trace(go.Scatter(
                x=df["round"],
                y=df[name] / 1_000,
                name=name,
                mode="lines",
                line=dict(color=color_map[name], width=2),
                hovertemplate=f"<b>{name}</b><br>Round %{{x}}<br>Profit: $%{{y:.1f}}K<extra></extra>",
            ))

    fig.update_layout(
        title=dict(text="Profit Growth Trend", x=0.5),
        xaxis_title="Round",
        yaxis_title="Profit ($K)",
        **_LAYOUT_DEFAULTS,
    )
    return fig


# ---------------------------------------------------------------------------
# 4. Demand Prediction Chart
# ---------------------------------------------------------------------------

def demand_prediction_chart(result: SimulationResult) -> go.Figure:
    """
    Plots total market demand with seasonality overlay.
    """
    df = result.demand_df()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=df["round"],
            y=df["total_demand"] / 1_000,
            name="Predicted Demand",
            mode="lines+markers",
            line=dict(color="#2196F3", width=2.5),
            marker=dict(size=5),
            hovertemplate="Round %{x}<br>Demand: %{y:.1f}K units<extra></extra>",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=df["round"],
            y=df["seasonality_factor"],
            name="Seasonality Factor",
            mode="lines",
            line=dict(color="#FF9800", width=1.5, dash="dot"),
            hovertemplate="Round %{x}<br>Seasonality: %{y:.3f}<extra></extra>",
        ),
        secondary_y=True,
    )

    fig.update_yaxes(title_text="Demand (K units)", secondary_y=False,
                     gridcolor="rgba(255,255,255,0.07)")
    fig.update_yaxes(title_text="Seasonality Factor", secondary_y=True)
    fig.update_layout(
        title=dict(text="Demand Prediction & Seasonality", x=0.5),
        xaxis_title="Round",
        **_LAYOUT_DEFAULTS,
    )
    return fig


# ---------------------------------------------------------------------------
# 5. Strategy Radar Chart
# ---------------------------------------------------------------------------

def strategy_radar(result: SimulationResult) -> go.Figure:
    """
    Radar chart comparing companies across key strategic dimensions
    (price inversion, marketing, quality, share, profit) in the final round.
    """
    final = result.final_states()
    categories = ["Price Competitiveness", "Marketing", "Quality",
                  "Market Share", "Profit Margin"]
    color_map = _color_map(result.company_names, result.company_colors)

    all_prices = [c.price for c in final]
    max_price = max(all_prices) if all_prices else 1.0
    max_marketing = max(c.marketing_budget for c in final) or 1.0
    max_profit = max(c.profit for c in final) or 1.0

    fig = go.Figure()
    for cs in final:
        price_comp = 1.0 - (cs.price / max_price)   # lower price = more competitive
        marketing_score = cs.marketing_budget / max_marketing
        quality_score = cs.product_quality
        share_score = cs.market_share
        profit_score = max(cs.profit, 0) / max(max_profit, 1)

        values = [price_comp, marketing_score, quality_score, share_score, profit_score]
        values_closed = values + [values[0]]   # close the polygon
        cats_closed = categories + [categories[0]]

        fig.add_trace(go.Scatterpolar(
            r=values_closed,
            theta=cats_closed,
            fill="toself",
            name=cs.name,
            line=dict(color=color_map.get(cs.name, "#FFFFFF")),
            opacity=0.6,
            hovertemplate=f"<b>{cs.name}</b><br>%{{theta}}: %{{r:.2f}}<extra></extra>",
        ))

    fig.update_layout(
        title=dict(text="Strategic Position Radar", x=0.5),
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(255,255,255,0.1)"),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
            bgcolor="rgba(0,0,0,0)",
        ),
        **_LAYOUT_DEFAULTS,
    )
    return fig


# ---------------------------------------------------------------------------
# 6. Feature Importance Bar Chart
# ---------------------------------------------------------------------------

def feature_importance_chart(importances: dict, title: str = "Feature Importance") -> go.Figure:
    """Horizontal bar chart for ML model feature importances."""
    sorted_items = sorted(importances.items(), key=lambda x: x[1])
    features, values = zip(*sorted_items) if sorted_items else ([], [])

    fig = go.Figure(go.Bar(
        y=list(features),
        x=list(values),
        orientation="h",
        marker=dict(
            color=list(values),
            colorscale="Blues",
            line=dict(color="rgba(0,0,0,0)", width=0),
        ),
        hovertemplate="%{y}: %{x:.3f}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Importance",
        **_LAYOUT_DEFAULTS,
    )
    return fig


# ---------------------------------------------------------------------------
# 7. Competition Index Over Time
# ---------------------------------------------------------------------------

def competition_index_chart(result: SimulationResult) -> go.Figure:
    """Area chart showing the HHI competition index over rounds."""
    rounds = [rr.round_number for rr in result.rounds]
    hhi = [rr.competition_index for rr in result.rounds]

    fig = go.Figure(go.Scatter(
        x=rounds,
        y=hhi,
        fill="tozeroy",
        mode="lines",
        line=dict(color="#9C27B0", width=2),
        fillcolor="rgba(156,39,176,0.15)",
        hovertemplate="Round %{x}<br>HHI: %{y:.3f}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text="Market Concentration (HHI)", x=0.5),
        xaxis_title="Round",
        yaxis_title="HHI Index",
        **_LAYOUT_DEFAULTS,
    )
    return fig
