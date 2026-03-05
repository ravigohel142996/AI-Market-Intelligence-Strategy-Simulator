"""
Export Panel – CSV and data download controls for the MarketMind AI dashboard.
"""

from __future__ import annotations

import streamlit as st

from analytics.market_metrics import MarketMetrics
from analytics.risk_metrics import RiskMetrics
from analytics.scenario_analyzer import ScenarioAnalyzer
from core.simulation_engine import SimulationResult


def render_export_panel(result: SimulationResult) -> None:
    """Render a section with CSV download buttons for all key data tables."""
    st.markdown('<p class="section-header">💾 Export Data</p>', unsafe_allow_html=True)

    metrics = MarketMetrics(result)
    risk = RiskMetrics(result)
    scenario = ScenarioAnalyzer(result)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        company_df = metrics.company_table()
        st.download_button(
            label="📋 Company Summary",
            data=company_df.to_csv(index=False),
            file_name="company_summary.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col2:
        shares_df = result.shares_df()
        st.download_button(
            label="📈 Market Shares",
            data=shares_df.to_csv(index=False),
            file_name="market_shares.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col3:
        risk_df = risk.risk_table()
        st.download_button(
            label="⚠️ Risk Metrics",
            data=risk_df.to_csv(index=False),
            file_name="risk_metrics.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col4:
        profits_df = result.profits_df()
        st.download_button(
            label="💰 Profit History",
            data=profits_df.to_csv(index=False),
            file_name="profit_history.csv",
            mime="text/csv",
            use_container_width=True,
        )
