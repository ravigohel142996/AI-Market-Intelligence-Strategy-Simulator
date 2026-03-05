"""
app.py – MarketMind AI Streamlit entry point.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from core.simulation_engine import SimulationEngine
from ui.controls import render_controls
from ui.dashboard import (
    configure_page,
    render_animated_section,
    render_charts,
    render_company_table,
    render_competitive_intelligence,
    render_empty_state,
    render_header,
    render_kpi_cards,
    render_model_insights,
    render_risk_section,
    render_simulation_info,
)
from ui.export_panel import render_export_panel


def main() -> None:
    configure_page()
    render_header()

    sim_cfg, market_cfg = render_controls()

    run_button = st.sidebar.button(
        "▶ Run Simulation",
        use_container_width=True,
        type="primary",
    )

    if run_button:
        with st.spinner("Training ML models and running simulation…"):
            engine = SimulationEngine(sim_cfg, market_cfg)
            result = engine.run()
        st.session_state["sim_result"] = result
        st.success(
            f"✅ Simulation complete – {sim_cfg.num_rounds} rounds, "
            f"{sim_cfg.num_companies} companies, "
            f"finished in {result.elapsed_seconds:.2f}s"
        )

    result = st.session_state.get("sim_result")

    if result is not None:
        render_kpi_cards(result)
        render_company_table(result)
        render_charts(result)
        render_animated_section(result)
        render_risk_section(result)
        render_competitive_intelligence(result)
        render_model_insights(result)
        render_export_panel(result)
        render_simulation_info(result)
    else:
        render_empty_state()


if __name__ == "__main__":
    main()
