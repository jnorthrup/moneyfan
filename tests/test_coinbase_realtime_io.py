import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "hrm"))

from coinbase_pipeline import CoinbaseInstruments, CoinbaseRealtime  # noqa: E402


def test_advanced_ticker_message_normalization():
    instruments = CoinbaseInstruments()
    realtime = CoinbaseRealtime(instruments=instruments, channels=["ticker"])
    realtime.subscribe(["BTC-USD"])

    msg = {
        "channel": "ticker",
        "sequence_num": 7,
        "events": [
            {
                "type": "snapshot",
                "tickers": [
                    {
                        "product_id": "BTC-USD",
                        "price": "102000.25",
                        "best_bid": "102000.20",
                        "best_ask": "102000.30",
                        "volume_24_h": "2500.0",
                        "time": "2026-02-16T12:00:00Z",
                    }
                ],
            }
        ],
    }

    realtime._process_message(msg)
    ticks = realtime.get_recent("BTC-USD", n=1)
    events = realtime.get_events(symbol="BTC-USD", channel="ticker", n=1)

    assert len(ticks) == 1
    assert ticks[0].close == 102000.25
    assert len(events) == 1
    assert events[0].sequence_num == 7


def test_sequence_gap_tracking():
    instruments = CoinbaseInstruments()
    realtime = CoinbaseRealtime(instruments=instruments, channels=["ticker"])
    realtime.subscribe(["ETH-USD"])

    msg1 = {
        "channel": "ticker",
        "sequence_num": 10,
        "events": [{"type": "update", "tickers": [{"product_id": "ETH-USD", "price": "3000"}]}],
    }
    msg2 = {
        "channel": "ticker",
        "sequence_num": 12,
        "events": [{"type": "update", "tickers": [{"product_id": "ETH-USD", "price": "3001"}]}],
    }

    realtime._process_message(msg1)
    realtime._process_message(msg2)

    assert len(realtime.sequence_gaps) == 1
    gap = realtime.sequence_gaps[0]
    assert gap["prev_sequence"] == 10
    assert gap["next_sequence"] == 12
