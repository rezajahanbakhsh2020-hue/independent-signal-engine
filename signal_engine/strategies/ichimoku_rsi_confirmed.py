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

    def __init__(
        self,
        tenkan_window: int = 9,
        kijun_window: int = 26,
        senkou_b_window: int = 52,
        displacement: int = 26,
        rsi_window: int = 14,
        rsi_midline: float = 50.0,
    ) -> None:
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
        # State records that price has approached from outside and traded into
        # the cloud. A confirmation is emitted only on the candle AFTER exit.
        state = "neutral"
        breakout_side = 0
        breakout_high = float("nan")
        breakout_low = float("nan")

        for i in range(len(df)):
            if pd.isna(cloud_top.iloc[i]) or pd.isna(cloud_bottom.iloc[i]) or pd.isna(rsi.iloc[i]):
                continue
            c = float(close.iloc[i])
            h = float(high.iloc[i])
            l = float(low.iloc[i])
            top = float(cloud_top.iloc[i])
            bottom = float(cloud_bottom.iloc[i])

            # First handle a pending breakout: the current candle is the
            # required next-candle confirmation, never the breakout candle.
            if breakout_side:
                if breakout_side == 1 and c > breakout_high and rsi.iloc[i] > self.rsi_midline:
                    out.iloc[i] = 1
                elif breakout_side == -1 and c < breakout_low and rsi.iloc[i] < self.rsi_midline:
                    out.iloc[i] = -1
                breakout_side = 0
                state = "neutral"
                # This candle can also start a fresh setup below.

            if c < bottom:
                state = "below"
            elif c > top:
                if state == "inside_from_below" and h >= bottom:
                    breakout_side = 1
                    breakout_high, breakout_low = h, l
                state = "above"
            else:
                # Candle overlaps or closes inside the cloud. Remember which
                # outside region price came from; ignore mere touches from above/below.
                if state == "below" and h >= bottom:
                    state = "inside_from_below"
                elif state == "above" and l <= top:
                    state = "inside_from_above"
                elif state == "inside_from_below" or state == "inside_from_above":
                    pass
                else:
                    state = "neutral"

            # Symmetric downside exit after trading into cloud from above.
            if c < bottom and state == "inside_from_above":
                breakout_side = -1
                breakout_high, breakout_low = h, l
                state = "below"

        return out
