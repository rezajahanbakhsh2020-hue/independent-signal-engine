from __future__ import annotations

import pandas as pd

from signal_engine import perpetual_dashboard as dashboard


def test_perpetual_keeps_actionable_strategy_event_when_risk_levels_are_unavailable(
    monkeypatch,
):
    now = pd.Timestamp.now(tz="UTC").floor("min")
    index = pd.date_range(end=now - pd.Timedelta(minutes=1), periods=40, freq="min")
    close = pd.Series([100.0 + i * 0.1 for i in range(len(index))], index=index)
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 10.0,
        },
        index=index,
    )
    frame.attrs.update(
        {
            "contract_symbol": "BTCUSDT",
            "source": "public:bitget-usdt-perpetual",
            "timestamp_unit": "milliseconds",
            "retrieval_time_utc": now.isoformat(),
            "validation_report": {"rows_out": len(frame)},
            "pagination_truncated": False,
        }
    )

    class OneStrategy:
        name = "Test_Reversal"

        def raw_direction(self, data):
            result = pd.Series(0, index=data.index, dtype="int64")
            result.iloc[-5] = 1       # actionable event: next candle exists
            result.iloc[-1] = -1      # pending event: no execution candle yet
            return result

    monkeypatch.setattr(dashboard, "STRATEGY_REGISTRY", {"Test_Reversal": OneStrategy})
    monkeypatch.setattr(dashboard, "fetch_range", lambda *args, **kwargs: frame)
    monkeypatch.setattr(
        dashboard,
        "latest_decision_report",
        lambda *args, **kwargs: {"decision": {"label": "NO TRADE", "reason": "test"}},
    )

    import signal_engine.dashboard as spot_dashboard

    monkeypatch.setattr(
        spot_dashboard, "_risk_levels_for_event", lambda *args, **kwargs: None
    )

    result = dashboard._poll("Bitget", "BTCUSDT", "1m", "Test_Reversal", 1)

    assert len(result["signals"]) == 1
    signal = result["signals"][0]
    assert signal["side"] == "BUY"
    assert signal["risk_status"] == "UNAVAILABLE"
    assert signal["entry"] is None
    assert signal["stop"] is None
    assert result["signal_diagnostics"] == {
        "strategy_events": 2,
        "actionable_events": 1,
        "pending_events": 1,
        "risk_levels_unavailable": 1,
    }


def test_perpetual_page_exposes_signal_diagnostics_and_risk_status():
    page = dashboard._page()
    assert 'id="signal-diagnostics"' in page
    assert "وضعیت سطوح ریسک" in page
    assert "سیگنال با سطوح ریسک ناموجود" in page
