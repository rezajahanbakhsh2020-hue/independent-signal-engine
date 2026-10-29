"""Strict OHLCV validation.

Principles:
- Data is either provably clean or explicitly rejected/documented.  We
  never silently fill prices, invent candles, or drop rows without
  recording them.
- Timestamps are normalized to UTC.
- "Punctual" x-candle deletions (e.g., zero-volume placeholder candles
  in crypto data) are never invented; gaps are documented in the report.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")
OPTIONAL_COLUMNS = ("timestamp", "time", "date", "datetime")


class DataValidationError(ValueError):
    """Raised when a dataset is fundamentally unusable (schema collapse)."""

    def __init__(self, message: str, report: "ValidationReport | None" = None):
        super().__init__(message)
        self.report = report


@dataclass
class ValidationReport:
    """Everything needed to audit what happened to a dataset."""

    n_rows_in: int = 0
    n_rows_out: int = 0
    n_duplicate_timestamps_removed: int = 0
    n_unsorted_repaired: int = 0
    n_rejected_rows: int = 0
    rejection_reasons: dict[str, int] = field(default_factory=dict)
    gaps: list[dict] = field(default_factory=list)
    start_utc: str | None = None
    end_utc: str | None = None
    is_monotonic: bool = True

    def reject(self, reason: str, count: int = 1) -> None:
        self.n_rejected_rows += count
        self.rejection_reasons[reason] = self.rejection_reasons.get(reason, 0) + count

    def as_dict(self) -> dict:
        return {
            "rows_in": self.n_rows_in,
            "rows_out": self.n_rows_out,
            "duplicates_removed": self.n_duplicate_timestamps_removed,
            "unsorted_repaired": self.n_unsorted_repaired,
            "rejected_rows": self.n_rejected_rows,
            "rejection_reasons": dict(self.rejection_reasons),
            "n_gaps": len(self.gaps),
            "gaps": self.gaps,
            "start_utc": self.start_utc,
            "end_utc": self.end_utc,
        }


def _coerce_numeric(df: pd.DataFrame, rep: ValidationReport) -> pd.DataFrame:
    """Coerce OHLCV columns to numeric, rejecting malformed rows explicitly."""
    out = df.copy()
    for col in REQUIRED_COLUMNS:
        coerced = pd.to_numeric(out[col], errors="coerce")
        bad_mask = coerced.isna() & out[col].notna()
        n_bad = int(bad_mask.sum())
        if n_bad:
            rep.reject(f"malformed_{col}", n_bad)
        out[col] = coerced
    return out


def _check_ohlc_relations(df: pd.DataFrame, rep: ValidationReport) -> pd.DataFrame:
    high = df["high"]
    low = df["low"]
    max_oc = df[["open", "close"]].max(axis=1)
    min_oc = df[["open", "close"]].min(axis=1)
    bad_high = high < max_oc
    bad_low = low > min_oc
    bad_cross = low > high
    bad = bad_high | bad_low | bad_cross
    n_bad = int(bad.sum())
    if n_bad:
        rep.reject("invalid_ohlc_relationship", n_bad)
        df = df[~bad]
    return df


def _check_finite(df: pd.DataFrame, rep: ValidationReport) -> pd.DataFrame:
    bad = pd.Series(False, index=df.index)
    for col in REQUIRED_COLUMNS:
        bad |= ~df[col].map(lambda x: isinstance(x, float) and math.isfinite(x))
    # also treat NaN in volume as rejection
    n_bad = int(bad.sum())
    if n_bad:
        rep.reject("non_finite_or_missing_value", n_bad)
        df = df[~bad]
    return df


def _check_volume(df: pd.DataFrame, rep: ValidationReport) -> pd.DataFrame:
    neg = df["volume"] < 0
    n_neg = int(neg.sum())
    if n_neg:
        rep.reject("negative_volume", n_neg)
        df = df[~neg]
    return df


def _detect_gaps(index: pd.DatetimeIndex, freq_td: pd.Timedelta) -> list[dict]:
    """Detect missing candles relative to the dominant spacing."""
    if len(index) < 2:
        return []
    gaps = []
    for i in range(1, len(index)):
        delta = index[i] - index[i - 1]
        if delta > freq_td:
            gaps.append(
                {
                    "after_utc": index[i - 1].isoformat(),
                    "next_utc": index[i].isoformat(),
                    "missing_candles_estimated": int(delta // freq_td) - 1,
                    "gap_seconds": int(delta.total_seconds()),
                }
            )
    return gaps


def dominant_spacing(index: pd.DatetimeIndex) -> pd.Timedelta:
    """Most frequent positive spacing in the index."""
    deltas = pd.Series(index[1:] - index[:-1])
    deltas = deltas[deltas > pd.Timedelta(0)]
    if deltas.empty:
        raise DataValidationError("Cannot infer spacing from fewer than two timestamps")
    return deltas.mode().iloc[0]


def validate_ohlcv(
    df: pd.DataFrame,
    strict_gap_policy: bool = True,
) -> tuple[pd.DataFrame, ValidationReport]:
    """Validate and normalize an OHLCV frame to a clean UTC-indexed dataset.

    Returns (clean_df, report).  Invalid rows are rejected and *recorded*;
    duplicate timestamps are removed (keeping the first) and counted;
    unsorted data is sorted and flagged.  Gaps are documented, never filled.
    """
    rep = ValidationReport()
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise DataValidationError("Input data must be a non-empty DataFrame")

    rep.n_rows_in = len(df)

    # 1. locate timestamp column or index.
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    ts_col = next(
        (c for c in df.columns if c in ("timestamp", "time", "date", "datetime")),
        None,
    )
    missing_required = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_required:
        raise DataValidationError(
            f"Missing required OHLCV columns: {', '.join(missing_required)}"
        )
    if ts_col is None and not isinstance(df.index, pd.DatetimeIndex):
        raise DataValidationError("No timestamp column and index is not a DatetimeIndex")
    if ts_col is not None:
        idx = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        n_bad_ts = int(idx.isna().sum())
        if n_bad_ts:
            rep.reject("malformed_timestamp", n_bad_ts)
            df = df[idx.notna()]
            idx = idx[idx.notna()]
        df.index = pd.DatetimeIndex(idx, name="timestamp")
        df = df.drop(columns=[ts_col])

    index = pd.DatetimeIndex(df.index, name="timestamp")
    if not index.empty and (index.tz is None):
        # Naive timestamps: assume they are already UTC (documented assumption).
        index = index.tz_localize("UTC")
    index = index.tz_convert("UTC")

    # 2. duplicates
    dup_mask = index.duplicated(keep="first")
    rep.n_duplicate_timestamps_removed = int(dup_mask.sum())
    if rep.n_duplicate_timestamps_removed:
        df = df[~dup_mask]
        index = index[~dup_mask]

    # 3. unsorted / repaired
    if not index.is_monotonic_increasing:
        order = index.argsort()
        rep.n_unsorted_repaired = int(
            (order != pd.RangeIndex(len(order))).sum()
        )
        df = df.iloc[order]
        index = index[order]

    df.index = pd.DatetimeIndex(index, name="timestamp")

    # 4. numeric coercion and row-level value checks.
    df = _coerce_numeric(df, rep)
    df = _check_finite(df, rep)
    df = _check_ohlc_relations(df, rep)
    df = _check_volume(df, rep)

    if df.empty:
        raise DataValidationError("All rows rejected during validation", report=rep)

    # 5. gaps (documented, never filled).
    spacing = dominant_spacing(df.index)
    rep.gaps = _detect_gaps(df.index, spacing)
    rep.is_monotonic = df.index.is_monotonic_increasing
    rep.start_utc = df.index[0].isoformat()
    rep.end_utc = df.index[-1].isoformat()
    rep.n_rows_out = len(df)

    if strict_gap_policy and rep.gaps:
        # Gaps are allowed but must be visible downstream; we do NOT infer
        # candles.  Large gaps make results harder to interpret, so they are
        # reported, not hidden.
        pass

    out = df[sorted(REQUIRED_COLUMNS)]
    return out, rep
