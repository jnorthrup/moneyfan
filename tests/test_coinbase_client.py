import os
from unittest.mock import MagicMock, patch

import pytest


class DummyResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


@patch.dict(os.environ, {}, clear=True)
def test_client_uses_default_base_url():
    import coinbase_auth

    with patch("coinbase_auth.requests.Session") as session_cls:
        session = MagicMock()
        session_cls.return_value = session

        client = coinbase_auth.CoinbaseClient()

        assert client.base_url == "https://api.coinbase.com"


@patch.dict(os.environ, {"COINBASE_API_URL": "https://example.coinbase.local"}, clear=True)
def test_client_reads_base_url_from_env():
    import coinbase_auth

    with patch("coinbase_auth.requests.Session") as session_cls:
        session = MagicMock()
        session_cls.return_value = session

        client = coinbase_auth.CoinbaseClient()

        assert client.base_url == "https://example.coinbase.local"


@patch.dict(os.environ, {}, clear=True)
def test_get_includes_bearer_header():
    import coinbase_auth

    response = DummyResponse(status_code=200, payload={"ok": True})

    with patch("coinbase_auth.generate_jwt_token", return_value="jwt-token"), patch(
        "coinbase_auth.requests.Session"
    ) as session_cls:
        session = MagicMock()
        session.request.return_value = response
        session_cls.return_value = session

        client = coinbase_auth.CoinbaseClient()
        out = client.get("/api/v3/brokerage/accounts")

        assert out is response
        _, kwargs = session.request.call_args
        assert kwargs["headers"]["Authorization"] == "Bearer jwt-token"


@patch.dict(os.environ, {}, clear=True)
def test_post_includes_bearer_header():
    import coinbase_auth

    response = DummyResponse(status_code=200, payload={"ok": True})

    with patch("coinbase_auth.generate_jwt_token", return_value="jwt-token"), patch(
        "coinbase_auth.requests.Session"
    ) as session_cls:
        session = MagicMock()
        session.request.return_value = response
        session_cls.return_value = session

        client = coinbase_auth.CoinbaseClient()
        out = client.post("/api/v3/brokerage/orders", json={"size": "1"})

        assert out is response
        _, kwargs = session.request.call_args
        assert kwargs["headers"]["Authorization"] == "Bearer jwt-token"


@patch.dict(os.environ, {}, clear=True)
def test_401_raises_authentication_error():
    import coinbase_auth

    response = DummyResponse(status_code=401, text="Unauthorized")

    with patch("coinbase_auth.generate_jwt_token", return_value="jwt-token"), patch(
        "coinbase_auth.requests.Session"
    ) as session_cls:
        session = MagicMock()
        session.request.return_value = response
        session_cls.return_value = session

        client = coinbase_auth.CoinbaseClient()

        with pytest.raises(coinbase_auth.AuthenticationError):
            client.get("/api/v3/brokerage/accounts")
