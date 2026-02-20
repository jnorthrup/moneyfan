import os
from decimal import Decimal
from unittest.mock import MagicMock, patch


SAMPLE_PRIVATE_KEY = """-----BEGIN EC PRIVATE KEY-----
MHcCAQEEINyDYF2xPWBIfsCbfya1JKCFwQZTq8k4oFWJ/iWcSLV0oAoGCCqGSM49
AwEHoUQDQgAEgO3wS/Q8UEscy9t8a1XsQLNW1IqrEJFZ6+2lDG5BYIfZ8DRShpuJ
iOkA31g7mg8GBjf9FrUmirJaAYtd02+IQw==
-----END EC PRIVATE KEY-----"""


class DummyResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


@patch.dict(
    os.environ,
    {"COINBASE_API_KEY_NAME": "test-key-name", "COINBASE_PRIVATE_KEY": SAMPLE_PRIVATE_KEY},
    clear=True,
)
def test_bot_initializes_with_credentials_present():
    from coinbase_live_trading import CoinbaseLiveTrading

    with patch("coinbase_live_trading.coinbase_auth.CoinbaseClient") as client_cls:
        bot = CoinbaseLiveTrading()

    assert bot is not None
    client_cls.assert_called_once()


@patch.dict(os.environ, {}, clear=True)
def test_simulation_mode_works_without_live_trading_set():
    from coinbase_live_trading import CoinbaseLiveTrading

    bot = CoinbaseLiveTrading()

    with patch.object(bot, "get_market_price", return_value=Decimal("50000")), patch(
        "coinbase_live_trading.coinbase_auth.place_market_order"
    ) as place_order_mock:
        result = bot.execute_trade(pair="BTC-USD", side="BUY", amount=Decimal("100"), order_type="maker")

    assert result["success"] is True
    place_order_mock.assert_not_called()


@patch.dict(
    os.environ,
    {"COINBASE_API_KEY_NAME": "test-key-name", "COINBASE_PRIVATE_KEY": SAMPLE_PRIVATE_KEY},
    clear=True,
)
def test_price_fetch_uses_coinbase_client_not_legacy_hmac():
    from coinbase_live_trading import CoinbaseLiveTrading

    fake_client = MagicMock()
    fake_client.get.return_value = DummyResponse({"price": "68629.51"})

    with patch("coinbase_live_trading.coinbase_auth.CoinbaseClient", return_value=fake_client):
        bot = CoinbaseLiveTrading()
        price = bot.get_market_price("BTC-USD")

    assert price == Decimal("68629.51")
    fake_client.get.assert_called_once_with("/api/v3/brokerage/products/BTC-USD")
