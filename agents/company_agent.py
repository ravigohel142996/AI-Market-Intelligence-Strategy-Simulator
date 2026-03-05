"""
Company Agent – the autonomous AI actor that represents a single company.

Each agent encapsulates its own:
  - current market position (price, marketing, quality, share, profit)
  - strategy engine (decision logic)
  - history of decisions and performance metrics
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from agents.strategy_engine import StrategyAction, StrategyEngine
from config import SimulationConfig
from core.market_environment import CompanyState, MarketEnvironment
from utils.helpers import clamp


@dataclass
class AgentDecision:
    """Record of a single round's decision and its outcome."""

    round_number: int
    action: StrategyAction
    price_before: float
    price_after: float
    marketing_before: float
    marketing_after: float
    quality_before: float
    quality_after: float
    market_share: float
    profit: float


class CompanyAgent:
    """
    Autonomous company agent combining state management with AI strategy.

    The agent observes the market, selects an action via its StrategyEngine,
    applies that action to its own attributes, and records the outcome.
    """

    def __init__(
        self,
        company_id: int,
        name: str,
        initial_price: float,
        initial_marketing: float,
        initial_quality: float,
        sim_cfg: SimulationConfig,
        rng: np.random.Generator,
    ) -> None:
        self.company_id = company_id
        self.name = name

        # Current state
        self.price = initial_price
        self.marketing_budget = initial_marketing
        self.product_quality = initial_quality
        self.brand_score: float = initial_quality * 0.5  # brand starts modest
        self.market_share: float = 0.0
        self.profit: float = 0.0

        self._sim_cfg = sim_cfg
        self._rng = rng
        self._strategy = StrategyEngine(company_id, sim_cfg)
        self._history: List[AgentDecision] = []
        self._last_profit: float = 0.0

    # ------------------------------------------------------------------
    # Decision loop
    # ------------------------------------------------------------------

    def decide(self, env: MarketEnvironment) -> None:
        """
        Observe the market environment and update internal attributes
        according to the selected strategy action.

        This mutates `self.price`, `self.marketing_budget`, and
        `self.product_quality` in-place.
        """
        state = self._build_state()
        action = self._strategy.select_action(state, env, self._rng)

        price_before = self.price
        marketing_before = self.marketing_budget
        quality_before = self.product_quality

        updates = StrategyEngine.apply_action(action, state, self._rng)
        self.price = updates["price"]
        self.marketing_budget = updates["marketing_budget"]
        self.product_quality = updates["product_quality"]

        self._history.append(AgentDecision(
            round_number=env.round_number,
            action=action,
            price_before=price_before,
            price_after=self.price,
            marketing_before=marketing_before,
            marketing_after=self.marketing_budget,
            quality_before=quality_before,
            quality_after=self.product_quality,
            market_share=self.market_share,
            profit=self.profit,
        ))

    def update_after_round(
        self, new_share: float, new_profit: float, env: MarketEnvironment
    ) -> None:
        """
        Apply post-round market dynamics updates and train the Q-function.

        Parameters
        ----------
        new_share  : market share assigned by the simulation engine
        new_profit : profit computed by market dynamics
        env        : the *updated* market environment
        """
        reward = (new_profit - self._last_profit) + (new_share - self.market_share) * 1_000

        self.market_share = new_share
        self.profit = new_profit
        self._last_profit = new_profit

        # Brand equity grows slowly with quality investment
        self.brand_score = clamp(
            self.brand_score * 0.95 + self.product_quality * 0.05, 0.0, 1.0
        )

        new_state = self._build_state()
        self._strategy.update_q(reward, new_state, env)

    # ------------------------------------------------------------------
    # State / history
    # ------------------------------------------------------------------

    def current_state(self) -> CompanyState:
        """Return an immutable snapshot of this agent's current state."""
        return self._build_state()

    def decision_history(self) -> List[AgentDecision]:
        return list(self._history)

    def last_action(self) -> Optional[StrategyAction]:
        return self._history[-1].action if self._history else None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_state(self) -> CompanyState:
        return CompanyState(
            company_id=self.company_id,
            name=self.name,
            price=self.price,
            marketing_budget=self.marketing_budget,
            product_quality=self.product_quality,
            brand_score=self.brand_score,
            market_share=self.market_share,
            profit=self.profit,
        )
