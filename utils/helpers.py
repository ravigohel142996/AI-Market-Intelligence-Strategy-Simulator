"""
Shared utility helpers used across the MarketMind AI codebase.
"""

from __future__ import annotations

import hashlib
import time
from functools import wraps
from typing import Any, Callable, Dict, List, TypeVar

import numpy as np
import pandas as pd

F = TypeVar("F", bound=Callable[..., Any])


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------

def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp *value* to the closed interval [lo, hi]."""
    return max(lo, min(hi, value))


def normalise(values: List[float]) -> List[float]:
    """
    Normalise a list of non-negative floats so they sum to 1.0.
    Returns a uniform distribution if the total is zero.
    """
    total = sum(values)
    if total <= 0:
        n = len(values)
        return [1.0 / n] * n
    return [v / total for v in values]


def pct_change(old: float, new: float) -> float:
    """Safe percentage change avoiding division by zero."""
    if old == 0:
        return 0.0
    return (new - old) / abs(old) * 100.0


# ---------------------------------------------------------------------------
# Dataframe helpers
# ---------------------------------------------------------------------------

def round_frame(df: pd.DataFrame, decimals: int = 4) -> pd.DataFrame:
    """Round all numeric columns to *decimals* places."""
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].round(decimals)
    return df


def wide_to_long(
    df: pd.DataFrame,
    id_vars: List[str],
    value_name: str = "value",
    var_name: str = "company",
) -> pd.DataFrame:
    """Convenience wrapper around pd.melt."""
    return df.melt(id_vars=id_vars, var_name=var_name, value_name=value_name)


# ---------------------------------------------------------------------------
# Timing decorator
# ---------------------------------------------------------------------------

def timer(func: F) -> F:
    """Log execution time of a function (prints to stdout)."""
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        print(f"[timer] {func.__qualname__} completed in {elapsed:.3f}s")
        return result
    return wrapper  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def stable_hash(obj: Any) -> int:
    """Return a deterministic integer hash suitable for seeding RNGs."""
    raw = str(obj).encode()
    return int(hashlib.sha256(raw).hexdigest(), 16) % (2**31)


def make_rng(seed: int) -> np.random.Generator:
    """Create a reproducible NumPy random generator."""
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def fmt_currency(value: float, symbol: str = "$") -> str:
    """Format a float as a human-readable currency string."""
    if abs(value) >= 1_000_000:
        return f"{symbol}{value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"{symbol}{value / 1_000:.1f}K"
    return f"{symbol}{value:.2f}"


def fmt_number(value: float) -> str:
    """Format a plain numeric quantity with K/M suffixes (no currency symbol)."""
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:.0f}"


def fmt_pct(value: float, decimals: int = 1) -> str:
    """Format a ratio (0-1) as a percentage string."""
    return f"{value * 100:.{decimals}f}%"


# ---------------------------------------------------------------------------
# Dict / collection helpers
# ---------------------------------------------------------------------------

def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge *override* into *base*, returning a new dict.
    Nested dicts are merged rather than replaced.
    """
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = deep_merge(result[key], val)
        else:
            result[key] = val
    return result
