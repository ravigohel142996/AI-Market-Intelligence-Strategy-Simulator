"""
Visualisations – Plotly chart builders for the MarketMind AI dashboard.

Every function accepts domain objects (SimulationResult, DataFrames) and
returns a fully configured `plotly.graph_objects.Figure` ready for rendering.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
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


# ---------------------------------------------------------------------------
# 8. Animated Market Share Race
# ---------------------------------------------------------------------------

def animated_market_share_race(result: SimulationResult) -> go.Figure:
    """
    Animated horizontal bar chart race showing market-share rankings
    across all simulation rounds.  Includes Play / Pause controls and
    a round slider.
    """
    shares_df = result.shares_df()
    company_names = result.company_names
    color_map = _color_map(company_names, result.company_colors)
    rounds = [int(r) for r in shares_df["round"].tolist()]

    def _frame_data(row: pd.Series) -> go.Bar:
        sorted_names = sorted(company_names, key=lambda n: row.get(n, 0))
        values = [row.get(n, 0) * 100 for n in sorted_names]
        colors = [color_map.get(n, "#FFFFFF") for n in sorted_names]
        return go.Bar(
            x=values,
            y=sorted_names,
            orientation="h",
            marker=dict(color=colors, line=dict(color="rgba(0,0,0,0)", width=0)),
            text=[f"{v:.1f}%" for v in values],
            textposition="outside",
            cliponaxis=False,
        )

    frames = [
        go.Frame(
            data=[_frame_data(shares_df.iloc[i])],
            name=str(rounds[i]),
            layout=go.Layout(
                title=dict(text=f"Market Share Race – Round {rounds[i]}", x=0.5)
            ),
        )
        for i in range(len(rounds))
    ]

    initial_data = _frame_data(shares_df.iloc[0])

    fig = go.Figure(data=[initial_data], frames=frames)

    fig.update_layout(
        title=dict(text=f"Market Share Race – Round {rounds[0]}", x=0.5),
        xaxis=dict(range=[0, 110], title="Market Share (%)",
                   gridcolor="rgba(255,255,255,0.07)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.07)"),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            direction="left",
            y=-0.18,
            x=0.5,
            xanchor="center",
            pad=dict(r=10, t=10),
            buttons=[
                dict(
                    label="▶ Play",
                    method="animate",
                    args=[
                        None,
                        {
                            "frame": {"duration": 700, "redraw": True},
                            "fromcurrent": True,
                            "transition": {"duration": 400, "easing": "cubic-in-out"},
                        },
                    ],
                ),
                dict(
                    label="⏸ Pause",
                    method="animate",
                    args=[
                        [None],
                        {
                            "frame": {"duration": 0, "redraw": False},
                            "mode": "immediate",
                            "transition": {"duration": 0},
                        },
                    ],
                ),
            ],
        )],
        sliders=[dict(
            active=0,
            steps=[
                dict(
                    method="animate",
                    args=[[str(r)], {
                        "frame": {"duration": 300, "redraw": True},
                        "mode": "immediate",
                        "transition": {"duration": 200},
                    }],
                    label=str(r),
                )
                for r in rounds
            ],
            x=0,
            y=-0.07,
            len=1.0,
            currentvalue=dict(
                prefix="Round: ",
                visible=True,
                xanchor="center",
                font=dict(size=13, color="#E0E0E0"),
            ),
            transition=dict(duration=200),
            bgcolor="rgba(255,255,255,0.05)",
            bordercolor="rgba(255,255,255,0.15)",
            tickcolor="rgba(255,255,255,0.3)",
        )],
        margin=dict(l=40, r=20, t=50, b=100),
        **{k: v for k, v in _LAYOUT_DEFAULTS.items() if k != "margin"},
    )
    return fig


# ---------------------------------------------------------------------------
# 9. Price–Quality Bubble Chart
# ---------------------------------------------------------------------------

def price_quality_bubble(result: SimulationResult) -> go.Figure:
    """
    Bubble chart: x = price, y = product quality,
    bubble size ∝ market share, colour = company.
    """
    final = result.final_states()
    color_map = _color_map(result.company_names, result.company_colors)

    fig = go.Figure()
    for cs in final:
        fig.add_trace(go.Scatter(
            x=[cs.price],
            y=[cs.product_quality],
            mode="markers+text",
            name=cs.name,
            text=[cs.name],
            textposition="top center",
            marker=dict(
                size=max(cs.market_share * 220, 12),
                color=color_map.get(cs.name, "#FFFFFF"),
                opacity=0.8,
                line=dict(color="rgba(255,255,255,0.5)", width=1),
            ),
            hovertemplate=(
                f"<b>{cs.name}</b><br>"
                "Price: $%{x:.2f}<br>"
                "Quality: %{y:.3f}<br>"
                f"Market Share: {cs.market_share * 100:.1f}%<extra></extra>"
            ),
        ))

    fig.update_layout(
        title=dict(text="Price–Quality Positioning (bubble = market share)", x=0.5),
        xaxis_title="Price ($)",
        yaxis_title="Product Quality (0–1)",
        showlegend=False,
        **_LAYOUT_DEFAULTS,
    )
    return fig


# ---------------------------------------------------------------------------
# 10. Volatility Bar Chart
# ---------------------------------------------------------------------------

def volatility_chart(risk_profiles: list) -> go.Figure:
    """
    Horizontal bar chart showing share volatility per company.
    *risk_profiles* is a list of ``CompanyRisk`` objects from RiskMetrics.
    """
    sorted_profiles = sorted(risk_profiles, key=lambda r: r.share_volatility)
    names = [r.name for r in sorted_profiles]
    volatilities = [r.share_volatility * 100 for r in sorted_profiles]
    stabilities = [r.stability_score for r in sorted_profiles]

    colors = [
        f"rgba({int(255 * (1 - s))}, {int(200 * s)}, {int(80 * s)}, 0.85)"
        for s in stabilities
    ]

    fig = go.Figure(go.Bar(
        x=volatilities,
        y=names,
        orientation="h",
        marker=dict(color=colors, line=dict(color="rgba(0,0,0,0)", width=0)),
        text=[f"{v:.2f}%" for v in volatilities],
        textposition="outside",
        hovertemplate="%{y}<br>Share Volatility: %{x:.2f}%<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text="Share Volatility by Company (lower = more stable)", x=0.5),
        xaxis_title="Share Std Dev (%)",
        **_LAYOUT_DEFAULTS,
    )
    return fig


# ---------------------------------------------------------------------------
# 11. Profit Correlation Heatmap
# ---------------------------------------------------------------------------

def correlation_heatmap(corr_matrix: pd.DataFrame) -> go.Figure:
    """Heatmap showing Pearson correlations between company profits."""
    names = list(corr_matrix.columns)
    values = corr_matrix.values

    fig = go.Figure(go.Heatmap(
        z=values,
        x=names,
        y=names,
        colorscale="RdBu",
        zmid=0,
        zmin=-1,
        zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in values],
        texttemplate="%{text}",
        hovertemplate="%{x} vs %{y}: %{z:.3f}<extra></extra>",
        colorbar=dict(
            title="Correlation",
            tickvals=[-1, -0.5, 0, 0.5, 1],
            ticktext=["-1.0", "-0.5", "0", "0.5", "1.0"],
        ),
    ))

    fig.update_layout(
        title=dict(text="Profit Correlation Between Companies", x=0.5),
        **_LAYOUT_DEFAULTS,
    )
    return fig


# ---------------------------------------------------------------------------
# 12. Marketing ROI Scatter
# ---------------------------------------------------------------------------

def marketing_roi_scatter(result: SimulationResult) -> go.Figure:
    """
    Scatter plot: x = marketing spend, y = average market share,
    coloured by company.  Helps identify which marketing budgets yield
    the best returns.
    """
    shares_df = result.shares_df()
    final = result.final_states()
    color_map = _color_map(result.company_names, result.company_colors)

    fig = go.Figure()
    for cs in final:
        if cs.name not in shares_df.columns:
            continue
        avg_share = float(shares_df[cs.name].mean()) * 100
        fig.add_trace(go.Scatter(
            x=[cs.marketing_budget / 1_000],
            y=[avg_share],
            mode="markers+text",
            name=cs.name,
            text=[cs.name],
            textposition="top center",
            marker=dict(
                size=16,
                color=color_map.get(cs.name, "#FFFFFF"),
                symbol="circle",
                line=dict(color="rgba(255,255,255,0.4)", width=1),
            ),
            hovertemplate=(
                f"<b>{cs.name}</b><br>"
                "Marketing: $%{x:.1f}K<br>"
                "Avg Share: %{y:.1f}%<extra></extra>"
            ),
        ))

    fig.update_layout(
        title=dict(text="Marketing Spend vs Average Market Share", x=0.5),
        xaxis_title="Marketing Budget ($K)",
        yaxis_title="Average Market Share (%)",
        showlegend=False,
        **_LAYOUT_DEFAULTS,
    )
    return fig
