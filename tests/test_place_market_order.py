import os
from unittest.mock import MagicMock, patch

import pytest


class DummyResponse:
    def __init__(self, payload=None):
        self._payload = payload or {}

    def json(self):
        return self._payload


@patch.dict(os.environ, {}, clear=True)
def test_place_market_order_no_live_trading_returns_none_and_logs():
    import coinbase_auth

    with patch("coinbase_auth.CoinbaseClient") as client_cls, patch("builtins.print") as print_mock:
        result = coinbase_auth.place_market_order("BTC-USD", "BUY", "0.001")

        assert result is None
        client_cls.assert_not_called()
        assert any("LIVE_TRADING" in str(call) for call in print_mock.call_args_list)


@patch.dict(os.environ, {"LIVE_TRADING": "true"}, clear=True)
def test_place_market_order_live_calls_orders_endpoint():
    import coinbase_auth

    fake_client = MagicMock()
    fake_client.post.return_value = DummyResponse(
        {"success": True, "success_response": {"order_id": "ord-123"}}
    )

    with patch("coinbase_auth.CoinbaseClient", return_value=fake_client):
        coinbase_auth.place_market_order("BTC-USD", "BUY", "0.001")

    fake_client.post.assert_called_once()
    call_args, call_kwargs = fake_client.post.call_args
    assert call_args[0] == "/api/v3/brokerage/orders"
    assert "json" in call_kwargs
    assert call_kwargs["json"]["product_id"] == "BTC-USD"
    assert call_kwargs["json"]["side"] == "BUY"


@patch.dict(os.environ, {"LIVE_TRADING": "true"}, clear=True)
def test_place_market_order_success_true_returns_order_object():
    import coinbase_auth

    order_obj = {"order_id": "ord-456", "status": "PENDING"}
    fake_client = MagicMock()
    fake_client.post.return_value = DummyResponse({"success": True, "success_response": order_obj})

    with patch("coinbase_auth.CoinbaseClient", return_value=fake_client):
        result = coinbase_auth.place_market_order("ETH-USD", "SELL", "0.01")

    assert result == order_obj


@patch.dict(os.environ, {"LIVE_TRADING": "true"}, clear=True)
def test_place_market_order_success_false_raises_order_error():
    import coinbase_auth

    fake_client = MagicMock()
    fake_client.post.return_value = DummyResponse(
        {"success": False, "error_response": {"message": "Insufficient funds"}}
    )

    with patch("coinbase_auth.CoinbaseClient", return_value=fake_client):
        with pytest.raises(coinbase_auth.OrderError) as exc_info:
            coinbase_auth.place_market_order("BTC-USD", "BUY", "1000")

    assert "error_response" in str(exc_info.value)
