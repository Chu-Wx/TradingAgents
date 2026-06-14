import pandas as pd
import pytest
import yfinance as yf

import tradingagents.dataflows.y_finance as yfin


@pytest.mark.unit
def test_get_yfin_calls_ticker_history_once(monkeypatch):
    calls = 0

    class FakeTicker:
        def history(self, **kwargs):
            nonlocal calls
            calls += 1
            return pd.DataFrame(
                {"Open": [1.0], "High": [1.0], "Low": [1.0], "Close": [1.0], "Volume": [1]},
                index=pd.to_datetime(["2026-06-12"]),
            )

    monkeypatch.setattr(yfin, "ticker_with_timeout", lambda symbol: FakeTicker())

    yfin.get_YFin_data_online("AAPL", "2026-06-12", "2026-06-12")

    assert calls == 1


@pytest.mark.integration
def test_real_yfinance_api_returns_aapl_history():
    data = yf.Ticker("AAPL").history(period="1d", timeout=30)

    assert not data.empty
