"""Shared deterministic fixtures."""
from __future__ import annotations

import pandas as pd
import pytest

from signal_engine.constants import TIMEFRAMES
from signal_engine.data.synthetic import synth_ohlcv


@pytest.fixture(params=TIMEFRAMES)
def synth_tf(request) -> pd.DataFrame:
    """Deterministic synthetic dataset for each canonical timeframe."""
    n = 600 if request.param not in ("1D",) else 500
    return synth_ohlcv(request.param, "2024-01-01", n, seed=42)


@pytest.fixture
def synth_1h() -> pd.DataFrame:
    return synth_ohlcv("1H", "2024-01-01", 600, seed=42)
