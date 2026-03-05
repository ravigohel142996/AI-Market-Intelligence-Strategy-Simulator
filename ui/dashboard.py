"""
Dashboard – assembles the Streamlit page layout.

This module contains only UI composition logic.  All data computation is
delegated to the analytics layer; all chart rendering is delegated to
ui.charts.
"""

from __future__ import annotations

import streamlit as st

from analytics.market_metrics import MarketMetrics
from analytics.risk_metrics import RiskMetrics
from analytics.scenario_analyzer import ScenarioAnalyzer
from core.simulation_engine import SimulationResult
from ui.charts import (
    render_animated_market_share_race,
    render_choice_feature_importance,
    render_competition_index,
    render_correlation_heatmap,
    render_demand_feature_importance,
    render_demand_prediction,
    render_market_share_evolution,
    render_market_share_pie,
    render_marketing_roi_scatter,
    render_price_quality_bubble,
    render_profit_trend,
    render_strategy_radar,
    render_volatility_chart,
)
from ui.export_panel import render_export_panel
from utils.helpers import fmt_currency, fmt_number, fmt_pct


# ---------------------------------------------------------------------------
# Page-level helpers
# ---------------------------------------------------------------------------

def configure_page() -> None:
    """Must be the very first Streamlit call in the app."""
    st.set_page_config(
        page_title="MarketMind AI",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_css()


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        /* ── General ── */
        .main { background-color: #0E1117; }
        h1, h2, h3 { font-family: 'Inter', sans-serif; }

        /* ── Keyframe animations ── */
        @keyframes fadeSlideUp {
            0%   { opacity: 0; transform: translateY(18px); }
            100% { opacity: 1; transform: translateY(0); }
        }
        @keyframes pulse-glow {
            0%   { box-shadow: 0 0 0 0 rgba(33, 150, 243, 0.0); }
            50%  { box-shadow: 0 0 16px 4px rgba(33, 150, 243, 0.35); }
            100% { box-shadow: 0 0 0 0 rgba(33, 150, 243, 0.0); }
        }
        @keyframes borderSlide {
            0%   { border-left-color: #2196F3; }
            33%  { border-left-color: #9C27B0; }
            66%  { border-left-color: #4CAF50; }
            100% { border-left-color: #2196F3; }
        }
        @keyframes shimmer {
            0%   { background-position: -200% center; }
            100% { background-position: 200% center; }
        }

        /* ── KPI Cards ── */
        .kpi-card {
            background: linear-gradient(135deg, #1E1E2E 0%, #252540 100%);
            border: 1px solid #3A3A5C;
            border-radius: 12px;
            padding: 20px 24px;
            text-align: center;
            animation: fadeSlideUp 0.6s ease-out both, pulse-glow 3s ease-in-out 0.8s infinite;
            transition: transform 0.2s ease, border-color 0.3s ease;
        }
        .kpi-card:hover {
            transform: translateY(-3px);
            border-color: #2196F3;
        }
        .kpi-value {
            font-size: 2rem;
            font-weight: 700;
            color: #FFFFFF;
            margin: 4px 0;
        }
        .kpi-label {
            font-size: 0.8rem;
            color: #9E9EC8;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }
        .kpi-delta {
            font-size: 0.85rem;
            color: #4CAF50;
            margin-top: 4px;
        }

        /* ── Section headings ── */
        .section-header {
            font-size: 1.2rem;
            font-weight: 600;
            color: #E0E0E0;
            border-left: 4px solid #2196F3;
            padding-left: 12px;
            margin: 24px 0 12px 0;
            animation: fadeSlideUp 0.5s ease-out both, borderSlide 6s linear infinite;
        }

        /* ── Risk badge ── */
        .risk-badge {
            display: inline-block;
            padding: 2px 10px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
        }
        .risk-low    { background: rgba(76, 175, 80, 0.2);  color: #4CAF50; }
        .risk-medium { background: rgba(255, 152, 0, 0.2);  color: #FF9800; }
        .risk-high   { background: rgba(244, 67, 54, 0.2);  color: #F44336; }

        /* ── Data table ── */
        .dataframe { border: none !important; }

        /* ── Animated hero banner ── */
        .hero-banner {
            background: linear-gradient(270deg, #0D47A1, #1565C0, #283593, #0D47A1);
            background-size: 600% 600%;
            animation: shimmer 8s ease infinite;
            border-radius: 12px;
            padding: 4px;
            margin-bottom: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------

def render_header() -> None:
    col_title, col_badge = st.columns([5, 1])
    with col_title:
        st.markdown(
            "<h1 style='margin-bottom:0'>🧠 MarketMind AI</h1>"
            "<p style='color:#9E9EC8;margin-top:4px'>"
            "Competitive Market Strategy Simulation Engine</p>",
            unsafe_allow_html=True,
        )
    with col_badge:
        st.markdown(
            "<div style='text-align:right;margin-top:16px'>"
            "<span style='background:#1E3A5F;color:#2196F3;padding:4px 10px;"
            "border-radius:20px;font-size:0.75rem;font-weight:600'>LIVE SIMULATION</span>"
            "</div>",
            unsafe_allow_html=True,
        )
    st.markdown("---")


def render_kpi_cards(result: SimulationResult) -> None:
    metrics = MarketMetrics(result)
    kpi = metrics.kpi_snapshot()

    st.markdown('<p class="section-header">📊 Market Overview</p>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-label">Total Market Value</div>'
            f'<div class="kpi-value">{fmt_currency(kpi.total_market_value)}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-label">Market Leader</div>'
            f'<div class="kpi-value">{kpi.market_leader_name}</div>'
            f'<div class="kpi-delta">{fmt_pct(kpi.market_leader_share)} share</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-label">Consumer Demand</div>'
            f'<div class="kpi-value">{fmt_number(kpi.total_demand)}</div>'
            f'<div class="kpi-label">units</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-label">Competition Index (HHI)</div>'
            f'<div class="kpi-value">{kpi.competition_index:.3f}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


def render_company_table(result: SimulationResult) -> None:
    st.markdown('<p class="section-header">🏢 Company Strategy Panel</p>', unsafe_allow_html=True)
    metrics = MarketMetrics(result)
    df = metrics.company_table()
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )


def render_charts(result: SimulationResult) -> None:
    st.markdown('<p class="section-header">📈 Market Share Evolution</p>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        render_market_share_evolution(result)
    with col2:
        render_market_share_pie(result)

    st.markdown('<p class="section-header">💰 Profit & Demand</p>', unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        render_profit_trend(result)
    with col4:
        render_demand_prediction(result)

    st.markdown('<p class="section-header">🎯 Strategic Positioning</p>', unsafe_allow_html=True)
    col5, col6 = st.columns(2)
    with col5:
        render_strategy_radar(result)
    with col6:
        render_competition_index(result)


def render_animated_section(result: SimulationResult) -> None:
    """Animated bar chart race with Play/Pause and round slider."""
    st.markdown(
        '<p class="section-header">🎬 Animated Market Replay</p>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Press **▶ Play** to watch the market-share race unfold round by round, "
        "or drag the slider to jump to any round."
    )
    render_animated_market_share_race(result)


def render_risk_section(result: SimulationResult) -> None:
    """Risk analysis: volatility, stability, and profit correlation."""
    st.markdown('<p class="section-header">⚠️ Risk Analysis</p>', unsafe_allow_html=True)

    risk = RiskMetrics(result)
    risk_df = risk.risk_table()

    st.dataframe(risk_df, use_container_width=True, hide_index=True)

    col_v, col_c = st.columns(2)
    with col_v:
        render_volatility_chart(result)
    with col_c:
        render_correlation_heatmap(result)


def render_competitive_intelligence(result: SimulationResult) -> None:
    """Competitive intelligence: price–quality positioning and marketing ROI."""
    st.markdown(
        '<p class="section-header">🔍 Competitive Intelligence</p>',
        unsafe_allow_html=True,
    )

    scenario = ScenarioAnalyzer(result)
    summary_df = scenario.competitive_summary_table()
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    col_a, col_b = st.columns(2)
    with col_a:
        render_price_quality_bubble(result)
    with col_b:
        render_marketing_roi_scatter(result)

    # Demand trend banner
    trend = scenario.demand_trend_summary()
    direction = "📈" if trend["trend_pct"] >= 0 else "📉"
    st.info(
        f"{direction} **Demand Trend:** first-half avg "
        f"**{fmt_number(trend['first_half_avg'])}** units → second-half avg "
        f"**{fmt_number(trend['second_half_avg'])}** units "
        f"(**{trend['trend_pct']:+.1f}%** change).  "
        f"Peak: **{fmt_number(trend['peak_demand'])}** | "
        f"Trough: **{fmt_number(trend['trough_demand'])}**"
    )


def render_model_insights(result: SimulationResult) -> None:
    st.markdown('<p class="section-header">🤖 ML Model Insights</p>', unsafe_allow_html=True)

    dm = result.demand_metrics
    cm = result.choice_metrics

    col_d, col_c = st.columns(2)
    with col_d:
        st.markdown("**Demand Predictor (Random Forest Regressor)**")
        m1, m2 = st.columns(2)
        m1.metric("R² Score", f"{dm.r2:.4f}")
        m2.metric("MAE", f"{dm.mae:,.0f} units")
        render_demand_feature_importance(result)

    with col_c:
        st.markdown("**Customer Choice Model (Random Forest Classifier)**")
        st.metric("Accuracy", f"{cm.accuracy:.4f}")
        render_choice_feature_importance(result)


def render_simulation_info(result: SimulationResult) -> None:
    with st.expander("ℹ️ Simulation Details", expanded=False):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"**Rounds completed:** {len(result.rounds)}")
            st.markdown(f"**Companies simulated:** {result.sim_cfg.num_companies}")
            st.markdown(f"**Random seed:** {result.sim_cfg.random_seed}")
        with col_b:
            st.markdown(f"**Market size:** {result.market_cfg.market_size:,.0f} units")
            st.markdown(f"**Consumer sensitivity:** {result.market_cfg.consumer_sensitivity}")
            st.markdown(f"**Elapsed time:** {result.elapsed_seconds:.2f}s")


def render_empty_state() -> None:
    st.markdown(
        """
        <div style="text-align:center;padding:80px 20px;color:#9E9EC8">
          <div style="font-size:4rem">🚀</div>
          <h2 style="color:#E0E0E0">Ready to Simulate</h2>
          <p>Configure the parameters in the sidebar and click <strong>▶ Run Simulation</strong>.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
