"""
Simulation Engine – orchestrates the full multi-round market simulation.

Responsibilities
----------------
1. Initialise companies and ML models.
2. For each round:
   a. Predict total market demand (ML model).
   b. Let each agent decide its strategy.
   c. Compute market shares and profits (market dynamics).
   d. Update agent states.
   e. Record round metrics.
3. Return a structured SimulationResult.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from agents.company_agent import CompanyAgent
from config import (
    DEFAULT_COMPANY_NAMES,
    COMPANY_COLORS,
    MarketConfig,
    ModelConfig,
    SimulationConfig,
)
from core.market_dynamics import MarketDynamics
from core.market_environment import CompanyState, MarketEnvironment
from models.customer_choice_model import CustomerChoiceModel, ChoiceModelMetrics
from models.demand_predictor import DemandPredictor, DemandModelMetrics
from utils.helpers import make_rng


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class RoundRecord:
    """All metrics captured for a single simulation round."""

    round_number: int
    total_demand: float
    seasonality_factor: float
    competition_index: float
    company_states: List[CompanyState]


@dataclass
class SimulationResult:
    """Full output of a completed simulation run."""

    sim_cfg: SimulationConfig
    market_cfg: MarketConfig
    rounds: List[RoundRecord]
    demand_metrics: DemandModelMetrics
    choice_metrics: ChoiceModelMetrics
    company_names: List[str]
    company_colors: List[str]
    elapsed_seconds: float

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def shares_df(self) -> pd.DataFrame:
        """
        Returns a DataFrame with shape (num_rounds, num_companies+1).
        Columns: 'round', company_name_1, …, company_name_n
        """
        records = []
        for rr in self.rounds:
            row: Dict = {"round": rr.round_number}
            for cs in rr.company_states:
                row[cs.name] = cs.market_share
            records.append(row)
        return pd.DataFrame(records)

    def profits_df(self) -> pd.DataFrame:
        """Returns a DataFrame with profit per company per round."""
        records = []
        for rr in self.rounds:
            row: Dict = {"round": rr.round_number}
            for cs in rr.company_states:
                row[cs.name] = cs.profit
            records.append(row)
        return pd.DataFrame(records)

    def demand_df(self) -> pd.DataFrame:
        """Returns a DataFrame with total_demand and seasonality per round."""
        return pd.DataFrame([
            {
                "round": rr.round_number,
                "total_demand": rr.total_demand,
                "seasonality_factor": rr.seasonality_factor,
            }
            for rr in self.rounds
        ])

    def final_states(self) -> List[CompanyState]:
        """Company states from the last round."""
        return self.rounds[-1].company_states if self.rounds else []


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class SimulationEngine:
    """
    Orchestrates the full market simulation.

    Usage
    -----
    engine = SimulationEngine(sim_cfg, market_cfg)
    result = engine.run()
    """

    def __init__(
        self,
        sim_cfg: SimulationConfig,
        market_cfg: MarketConfig = MarketConfig(),
        model_cfg: ModelConfig = ModelConfig(),
    ) -> None:
        self._sim_cfg = sim_cfg
        self._market_cfg = market_cfg
        self._model_cfg = model_cfg

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> SimulationResult:
        """Execute the simulation and return the complete result."""
        t0 = time.perf_counter()
        rng = make_rng(self._sim_cfg.random_seed)

        # 1. Train ML models
        demand_model = DemandPredictor(
            self._market_cfg, self._model_cfg, self._sim_cfg.random_seed
        )
        demand_metrics = demand_model.train()

        choice_model = CustomerChoiceModel(
            self._sim_cfg.num_companies, self._model_cfg, self._sim_cfg.random_seed
        )
        choice_metrics = choice_model.train()

        # 2. Initialise market dynamics and agents
        dynamics = MarketDynamics(self._market_cfg, choice_model)
        agents = self._create_agents(rng)

        rounds: List[RoundRecord] = []

        # Build initial environment (equal shares, zero profit)
        env = self._build_initial_env(agents, round_number=0)
        env.refresh_derived_metrics()

        # 3. Simulation loop
        for round_num in range(1, self._sim_cfg.num_rounds + 1):
            env.round_number = round_num
            env.seasonality_factor = env.compute_seasonality()

            # 3a. Demand prediction
            avg_price = np.mean([a.price for a in agents])
            median_competitor_price = float(np.median([a.price for a in agents]))
            total_demand = demand_model.predict(
                price=avg_price,
                marketing_budget=float(np.mean([a.marketing_budget for a in agents])),
                product_quality=float(np.mean([a.product_quality for a in agents])),
                seasonality=env.seasonality_factor,
                competitor_price=median_competitor_price,
            )
            env.total_demand = max(total_demand, 0.0)

            # 3b. Agent decisions
            for agent in agents:
                agent.decide(env)

            # Refresh env with post-decision states
            env.companies = [a.current_state() for a in agents]

            # 3c. Market dynamics
            outcomes = dynamics.compute_round_outcomes(env.companies, env.total_demand)

            # 3d. Update agents
            for agent, (new_share, new_profit) in zip(agents, outcomes):
                agent.update_after_round(new_share, new_profit, env)

            # Refresh env with updated states
            env.companies = [a.current_state() for a in agents]
            env.competition_index = env.compute_competition_index()

            # 3e. Record metrics
            rounds.append(RoundRecord(
                round_number=round_num,
                total_demand=env.total_demand,
                seasonality_factor=env.seasonality_factor,
                competition_index=env.competition_index,
                company_states=list(env.companies),
            ))

        elapsed = time.perf_counter() - t0

        colors = (COMPANY_COLORS * ((self._sim_cfg.num_companies // len(COMPANY_COLORS)) + 1))[
            :self._sim_cfg.num_companies
        ]

        return SimulationResult(
            sim_cfg=self._sim_cfg,
            market_cfg=self._market_cfg,
            rounds=rounds,
            demand_metrics=demand_metrics,
            choice_metrics=choice_metrics,
            company_names=[a.name for a in agents],
            company_colors=colors,
            elapsed_seconds=elapsed,
        )

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _create_agents(self, rng: np.random.Generator) -> List[CompanyAgent]:
        n = self._sim_cfg.num_companies
        names = DEFAULT_COMPANY_NAMES[:n]
        # Pad with generic names if more companies requested than names defined
        while len(names) < n:
            names.append(f"Company{len(names) + 1}")

        lo_p, hi_p = self._sim_cfg.price_range
        lo_m, hi_m = self._sim_cfg.marketing_range
        lo_q, hi_q = self._sim_cfg.quality_range

        agents: List[CompanyAgent] = []
        for i in range(n):
            seed_i = int(rng.integers(0, 2**30))
            agent_rng = make_rng(seed_i)
            agents.append(CompanyAgent(
                company_id=i,
                name=names[i],
                initial_price=float(rng.uniform(lo_p, hi_p)),
                initial_marketing=float(rng.uniform(lo_m, hi_m)),
                initial_quality=float(rng.uniform(lo_q, hi_q)),
                sim_cfg=self._sim_cfg,
                rng=agent_rng,
            ))
        return agents

    def _build_initial_env(
        self, agents: List[CompanyAgent], round_number: int
    ) -> MarketEnvironment:
        n = len(agents)
        equal_share = 1.0 / n if n > 0 else 0.0
        for agent in agents:
            agent.market_share = equal_share

        return MarketEnvironment(
            round_number=round_number,
            market_cfg=self._market_cfg,
            companies=[a.current_state() for a in agents],
            total_demand=0.0,
        )
