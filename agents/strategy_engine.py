"""
Strategy Engine – decision logic for company AI agents.

The strategy engine encapsulates all reasoning about *what action to take*
given the current market environment. It is intentionally decoupled from
the CompanyAgent so strategies can be swapped independently of agent state.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Optional

import numpy as np

from config import SimulationConfig
from core.market_environment import CompanyState, MarketEnvironment
from utils.helpers import clamp


class StrategyAction(Enum):
    HOLD = auto()
    INCREASE_PRICE = auto()
    DECREASE_PRICE = auto()
    INCREASE_MARKETING = auto()
    DECREASE_MARKETING = auto()
    IMPROVE_QUALITY = auto()
    AGGRESSIVE_DISCOUNT = auto()


# Magnitude of each action (expressed as a fraction of current value)
_ACTION_MAGNITUDE = {
    StrategyAction.HOLD: 0.0,
    StrategyAction.INCREASE_PRICE: 0.07,
    StrategyAction.DECREASE_PRICE: 0.07,
    StrategyAction.INCREASE_MARKETING: 0.10,
    StrategyAction.DECREASE_MARKETING: 0.10,
    StrategyAction.IMPROVE_QUALITY: 0.05,
    StrategyAction.AGGRESSIVE_DISCOUNT: 0.15,
}


class StrategyEngine:
    """
    ε-greedy strategy selection with gradient-following exploitation.

    The engine maintains a simple Q-table keyed by (relative_share_bucket,
    relative_profit_bucket) → action score.  After each round, the Q-value
    of the chosen action is updated via a Bellman-style rule.
    """

    _ACTIONS = list(StrategyAction)
    _NUM_BUCKETS = 5   # quantise share/profit into 5 buckets each

    def __init__(self, company_id: int, sim_cfg: SimulationConfig) -> None:
        self._company_id = company_id
        self._learning_rate = sim_cfg.strategy_learning_rate
        self._exploration_rate = sim_cfg.exploration_rate

        # Q-table: (share_bucket, profit_bucket) → score per action
        shape = (self._NUM_BUCKETS, self._NUM_BUCKETS, len(self._ACTIONS))
        self._q_table: np.ndarray = np.zeros(shape)

        self._last_action: Optional[StrategyAction] = None
        self._last_state_key: Optional[tuple] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_action(
        self, agent: CompanyState, env: MarketEnvironment, rng: np.random.Generator
    ) -> StrategyAction:
        """
        Choose the next strategy action using ε-greedy policy.

        Parameters
        ----------
        agent : current state of this company
        env   : full market environment (used to observe competitors)
        rng   : reproducible random generator
        """
        state_key = self._encode_state(agent, env)

        if rng.random() < self._exploration_rate:
            action = rng.choice(self._ACTIONS)
        else:
            q_values = self._q_table[state_key]
            action = self._ACTIONS[int(np.argmax(q_values))]

        self._last_action = action
        self._last_state_key = state_key
        return action

    def update_q(
        self,
        reward: float,
        new_agent: CompanyState,
        new_env: MarketEnvironment,
    ) -> None:
        """
        Bellman update: Q(s,a) ← Q(s,a) + α·[reward + γ·max_a' Q(s',a') – Q(s,a)]

        Parameters
        ----------
        reward    : scalar reward (profit delta + market-share delta)
        new_agent : agent state *after* the action was applied
        new_env   : market environment *after* the action was applied
        """
        if self._last_state_key is None or self._last_action is None:
            return

        gamma = 0.9
        action_idx = self._ACTIONS.index(self._last_action)
        new_state_key = self._encode_state(new_agent, new_env)

        current_q = self._q_table[self._last_state_key][action_idx]
        future_q = float(np.max(self._q_table[new_state_key]))
        td_target = reward + gamma * future_q
        self._q_table[self._last_state_key][action_idx] += (
            self._learning_rate * (td_target - current_q)
        )

    # ------------------------------------------------------------------
    # State-action application
    # ------------------------------------------------------------------

    @staticmethod
    def apply_action(
        action: StrategyAction,
        agent: CompanyState,
        rng: np.random.Generator,
    ) -> dict:
        """
        Apply *action* to *agent* and return a dict of updated attributes.

        The caller is responsible for writing these back to the agent.
        """
        mag = _ACTION_MAGNITUDE[action]

        new_price = agent.price
        new_marketing = agent.marketing_budget
        new_quality = agent.product_quality

        if action == StrategyAction.INCREASE_PRICE:
            new_price = agent.price * (1.0 + mag)
        elif action == StrategyAction.DECREASE_PRICE:
            new_price = agent.price * (1.0 - mag)
        elif action == StrategyAction.AGGRESSIVE_DISCOUNT:
            new_price = agent.price * (1.0 - mag)
        elif action == StrategyAction.INCREASE_MARKETING:
            new_marketing = agent.marketing_budget * (1.0 + mag)
        elif action == StrategyAction.DECREASE_MARKETING:
            new_marketing = agent.marketing_budget * (1.0 - mag)
        elif action == StrategyAction.IMPROVE_QUALITY:
            new_quality = clamp(agent.product_quality + mag, 0.0, 1.0)

        return {
            "price": clamp(new_price, 10.0, 500.0),
            "marketing_budget": clamp(new_marketing, 1_000.0, 500_000.0),
            "product_quality": clamp(new_quality, 0.0, 1.0),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_state(
        self, agent: CompanyState, env: MarketEnvironment
    ) -> tuple:
        """Discretise continuous state into (share_bucket, profit_bucket)."""
        share_bucket = self._quantise(agent.market_share, 0.0, 1.0)
        max_profit = max((c.profit for c in env.companies), default=1.0)
        profit_normalised = agent.profit / max_profit if max_profit > 0 else 0.0
        profit_bucket = self._quantise(profit_normalised, 0.0, 1.0)
        return (share_bucket, profit_bucket)

    def _quantise(self, value: float, lo: float, hi: float) -> int:
        """Map *value* in [lo, hi] to bucket index 0 … NUM_BUCKETS-1."""
        span = hi - lo
        if span == 0:
            return 0
        ratio = clamp((value - lo) / span, 0.0, 1.0)
        return min(int(ratio * self._NUM_BUCKETS), self._NUM_BUCKETS - 1)
