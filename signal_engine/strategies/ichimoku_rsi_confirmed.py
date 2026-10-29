"""Ichimoku cloud breakout with RSI and next-candle confirmation.

The cloud visible at candle i is built from Senkou spans calculated 26 candles
earlier (shifted back onto the candle where that cloud is actually plotted).
A breakout requires price to approach from outside, trade into the cloud, and
close beyond its opposite edge. The following completed candle must confirm by
closing beyond the breakout candle's high/low and RSI(14) must agree (>50 long,
<50 short). The backtester then executes at the next candle open.
"""
from __future__ import annotations

import pandas as pd

from signal_engine.strategies.base import Strategy


class IchimokuRsiConfirmedStrategy(Strategy):
    name = "ichimoku_rsi_confirmed"

    def __init__(self, tenkan_window: int = 9, kijun_window: int = 26,
                 senkou_b_window: int = 52, displacement: int = 26,
                 rsi_window: int = 14, rsi_midline: float = 50.0) -> None:
        if min(tenkan_window, kijun_window, senkou_b_window, rsi_window) < 2:
            raise ValueError("indicator windows must be >= 2")
        if displacement < 0:
            raise ValueError("displacement must be >= 0")
        if not 0 < rsi_midline < 100:
            raise ValueError("rsi_midline must be between 0 and 100")
        self.tenkan_window = tenkan_window
        self.kijun_window = kijun_window
        self.senkou_b_window = senkou_b_window
        self.displacement = displacement
        self.rsi_window = rsi_window
        self.rsi_midline = float(rsi_midline)

    @staticmethod
    def _midpoint(df: pd.DataFrame, window: int) -> pd.Series:
        hi = df["high"].rolling(window=window, min_periods=window).max()
        lo = df["low"].rolling(window=window, min_periods=window).min()
        return (hi + lo) / 2.0

    def raw_direction(self, df: pd.DataFrame) -> pd.Series:
        high, low, close = df["high"], df["low"], df["close"]
        tenkan = self._midpoint(df, self.tenkan_window)
        kijun = self._midpoint(df, self.kijun_window)
        span_a = ((tenkan + kijun) / 2.0).shift(self.displacement)
        span_b = self._midpoint(df, self.senkou_b_window).shift(self.displacement)
        cloud_top = pd.concat([span_a, span_b], axis=1).max(axis=1)
        cloud_bottom = pd.concat([span_a, span_b], axis=1).min(axis=1)

        delta = close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / self.rsi_window, adjust=False,
                                       min_periods=self.rsi_window).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / self.rsi_window, adjust=False,
                                          min_periods=self.rsi_window).mean()
        rs = gain / loss.where(loss != 0)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        rsi = rsi.mask((loss == 0) & (gain > 0), 100.0)
        rsi = rsi.mask((loss == 0) & (gain == 0), 50.0)

        out = pd.Series(0, index=df.index, dtype="int64")
        state = "neutral"
        pending_side = 0
        pending_high = pending_low = float("nan")

        for i in range(len(df)):
            if pd.isna(cloud_top.iloc[i]) or pd.isna(cloud_bottom.iloc[i]) or pd.isna(rsi.iloc[i]):
                continue
            c, h, l = float(close.iloc[i]), float(high.iloc[i]), float(low.iloc[i])
            top, bottom = float(cloud_top.iloc[i]), float(cloud_bottom.iloc[i])

            # Confirmation belongs to the candle after the cloud-exit candle.
            if pending_side:
                if pending_side == 1 and c > pending_high and rsi.iloc[i] > self.rsi_midline:
                    out.iloc[i] = 1
                elif pending_side == -1 and c < pending_low and rsi.iloc[i] < self.rsi_midline:
                    out.iloc[i] = -1
                pending_side = 0

            # Detect the cloud exit only if the price previously entered from
            # the opposite side. Do not use future candles or shifted-forward spans.
            if c > top and state == "inside_from_below":
                pending_side, pending_high, pending_low = 1, h, l
                state = "above"
            elif c < bottom and state == "inside_from_above":
                pending_side, pending_high, pending_low = -1, h, l
                state = "below"
            elif c < bottom:
                state = "below"
            elif c > top:
                state = "above"
            else:
                if state == "below" and h >= bottom:
                    state = "inside_from_below"
                elif state == "above" and l <= top:
                    state = "inside_from_above"
                elif state not in ("inside_from_below", "inside_from_above"):
                    state = "neutral"

        return out
