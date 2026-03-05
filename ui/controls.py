"""
Sidebar controls – renders all Streamlit sidebar widgets and returns
the user-selected simulation configuration.
"""

from __future__ import annotations

import streamlit as st

from config import MarketConfig, SimulationConfig


def render_controls() -> tuple[SimulationConfig, MarketConfig]:
    """
    Render sidebar inputs and return (SimulationConfig, MarketConfig).

    All default values are pulled from the config dataclasses so the UI
    stays in sync with the rest of the system.
    """
    sim_defaults = SimulationConfig()
    mkt_defaults = MarketConfig()

    st.sidebar.title("⚙️ Simulation Controls")
    st.sidebar.markdown("---")

    # ------------------------------------------------------------------
    # Simulation parameters
    # ------------------------------------------------------------------
    st.sidebar.subheader("Simulation Parameters")

    num_companies = st.sidebar.slider(
        "Number of Companies",
        min_value=2,
        max_value=8,
        value=sim_defaults.num_companies,
        step=1,
        help="How many competing company AI agents to initialise.",
    )

    num_rounds = st.sidebar.slider(
        "Simulation Rounds",
        min_value=5,
        max_value=50,
        value=sim_defaults.num_rounds,
        step=5,
        help="Total number of rounds to simulate.",
    )

    random_seed = st.sidebar.number_input(
        "Random Seed",
        min_value=0,
        max_value=99_999,
        value=sim_defaults.random_seed,
        step=1,
        help="Seed for reproducibility. Change to explore different outcomes.",
    )

    st.sidebar.markdown("---")

    # ------------------------------------------------------------------
    # Market parameters
    # ------------------------------------------------------------------
    st.sidebar.subheader("Market Parameters")

    market_size = st.sidebar.number_input(
        "Market Size (units)",
        min_value=10_000,
        max_value=10_000_000,
        value=int(mkt_defaults.market_size),
        step=50_000,
        help="Total addressable market in units.",
    )

    consumer_sensitivity = st.sidebar.slider(
        "Consumer Price Sensitivity",
        min_value=0.1,
        max_value=1.0,
        value=mkt_defaults.consumer_sensitivity,
        step=0.05,
        help="How strongly consumers respond to price changes (0=none, 1=max).",
    )

    price_elasticity = st.sidebar.slider(
        "Price Elasticity",
        min_value=0.5,
        max_value=3.0,
        value=mkt_defaults.price_elasticity,
        step=0.1,
        help="Elasticity of demand with respect to price.",
    )

    brand_loyalty = st.sidebar.slider(
        "Brand Loyalty",
        min_value=0.0,
        max_value=1.0,
        value=mkt_defaults.brand_loyalty,
        step=0.05,
        help="Stickiness to incumbent brands (0=none, 1=full loyalty).",
    )

    competition_intensity = st.sidebar.slider(
        "Competition Intensity",
        min_value=0.0,
        max_value=1.0,
        value=mkt_defaults.competition_intensity,
        step=0.05,
        help="Overall aggressiveness of the competitive environment.",
    )

    st.sidebar.markdown("---")

    sim_cfg = SimulationConfig(
        num_companies=int(num_companies),
        num_rounds=int(num_rounds),
        random_seed=int(random_seed),
    )

    market_cfg = MarketConfig(
        market_size=float(market_size),
        consumer_sensitivity=float(consumer_sensitivity),
        price_elasticity=float(price_elasticity),
        brand_loyalty=float(brand_loyalty),
        competition_intensity=float(competition_intensity),
    )

    return sim_cfg, market_cfg
