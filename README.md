# 🧠 MarketMind AI – Competitive Market Strategy Simulation Engine

An AI-powered market simulation platform where multiple companies compete in a
simulated environment driven by machine learning demand forecasting and
reinforcement-style strategy agents.

---

## Project Overview

MarketMind AI combines classical machine learning (Random Forest demand
prediction, customer choice modelling) with AI strategy agents (ε-greedy
Q-learning) to simulate how companies evolve competitive strategies over time.

The entire system is surfaced through a Streamlit dashboard with interactive
controls, Plotly visualisations, and real-time ML model insights.

---

## Features

- **Multi-company AI agents** that autonomously select pricing, marketing, and quality strategies each round
- **ML demand prediction** – Random Forest Regressor trained on synthetic market data
- **Customer choice model** – Random Forest Classifier estimating per-company consumer preference
- **Market dynamics engine** – blends ML-based shares with attraction-gravity economics
- **Enterprise dashboard** – KPI cards, tabular strategy panel, and 7 Plotly charts
- **Fully configurable** – all parameters adjustable from the sidebar without touching code
- **Reproducible** – deterministic results via configurable random seed

---

## Architecture

```
marketmind-ai/
├── app.py                      # Streamlit entry point
├── config.py                   # Central configuration (no hardcoded values)
│
├── core/
│   ├── market_environment.py   # Market state snapshot & derived metrics
│   ├── simulation_engine.py    # Multi-round orchestration loop
│   └── market_dynamics.py      # Share & profit computation
│
├── agents/
│   ├── company_agent.py        # Autonomous AI company actor
│   └── strategy_engine.py      # ε-greedy Q-learning strategy selector
│
├── models/
│   ├── demand_predictor.py     # Random Forest demand model
│   └── customer_choice_model.py# Random Forest choice classifier
│
├── analytics/
│   ├── market_metrics.py       # KPI aggregation & summaries
│   └── visualizations.py       # Plotly chart builders
│
├── ui/
│   ├── dashboard.py            # Page layout & section renderers
│   ├── controls.py             # Sidebar widget definitions
│   └── charts.py               # Streamlit chart rendering wrappers
│
└── utils/
    ├── data_generator.py       # Synthetic training data generation
    └── helpers.py              # Shared utilities (clamp, normalise, fmt…)
```

---

## Tech Stack

| Layer | Library |
|---|---|
| UI | Streamlit >= 1.32 |
| ML | scikit-learn >= 1.4 |
| Data | Pandas >= 2.1, NumPy >= 1.26 |
| Charts | Plotly >= 5.19 |

---

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/ravigohel142996/AI-Market-Intelligence-Strategy-Simulator.git
cd AI-Market-Intelligence-Strategy-Simulator

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the dashboard
streamlit run app.py
```

Open http://localhost:8501 in your browser, configure parameters in the
sidebar, and click **Run Simulation**.

---

## Dashboard Sections

| Section | Description |
|---|---|
| Market Overview | KPI cards: Total Market Value, Market Leader, Consumer Demand, HHI |
| Company Strategy Panel | Live table: price, marketing, quality, share, profit |
| Market Share Evolution | Line chart + pie chart |
| Profit & Demand | Profit growth trend + demand prediction with seasonality |
| Strategic Positioning | Radar chart + competition concentration index |
| ML Model Insights | R2, MAE, feature importance for both models |

---

## Future Improvements

- **Reinforcement learning agents** with neural-network policy networks (PPO/DQN)
- **External data ingestion** – plug in real market data via API connectors
- **Multi-market simulation** – model geographic or segment-specific sub-markets
- **Agent communication** – coalitions, price-signalling, and collusion detection
- **Scenario analysis** – shock events (recession, new entrant, regulation change)
- **Export** – CSV / PDF report generation from any simulation run
