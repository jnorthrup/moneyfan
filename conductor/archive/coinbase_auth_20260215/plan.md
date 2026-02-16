# Plan: Fix Coinbase Advanced Trade Authentication and Enable Live Order Placement

Track ID: `coinbase_auth_20260215`

---

## Phase 1: JWT Authentication Module [checkpoint: e6724c5]

- [x] Task: Write failing tests for JWT token generation (Red) [aac51f7]
    - [x] Test that `generate_jwt(method, path)` returns a valid JWT string
    - [x] Test that JWT header contains `alg: ES256` and `kid: <api_key_name>`
    - [x] Test that JWT claims contain `sub`, `iss`, `nbf`, `exp`, `uri`
    - [x] Test that `exp` is within 120 seconds of `nbf`
    - [x] Test that missing env vars raise a clear `EnvironmentError`
- [x] Task: Implement `coinbase_auth.py` — JWT generation module (Green) [98833ad]
    - [x] Load `COINBASE_API_KEY_NAME` and `COINBASE_PRIVATE_KEY` from env / `.env`
    - [x] Parse EC private key (support raw PEM and CDP SDK format)
    - [x] Sign JWT with ES256 using `PyJWT` or `python-jose`
    - [x] Expose `generate_jwt(method: str, path: str) -> str`
- [x] Task: Conductor - User Manual Verification 'Phase 1' (Protocol in workflow.md)

---

## Phase 2: Authenticated HTTP Client

- [x] Task: Write failing tests for `CoinbaseClient` (Red) [7cc1cba]
    - [x] Test that GET request includes `Authorization: Bearer <token>` header
    - [x] Test that POST request includes `Authorization: Bearer <token>` header
    - [x] Test that HTTP 401 response raises `AuthenticationError` with clear message
    - [x] Test that client reads base URL from env (`COINBASE_API_URL`, default `https://api.coinbase.com`)
- [x] Task: Implement `CoinbaseClient` class (Green) [7cc1cba]
    - [x] Wrap `requests.Session` with auto JWT injection
    - [x] Implement `get(path, **kwargs)` and `post(path, json, **kwargs)`
    - [x] Handle 401 with descriptive error
- [x] Task: Verify authenticated endpoints return 200 OK
    - [x] `GET /api/v3/brokerage/accounts` → 200 + non-empty account list
    - [x] `GET /api/v3/brokerage/products/BTC-USD` → 200 + product data
    - [x] `GET /api/v3/brokerage/orders/historical/batch` → 200
- [x] Task: Conductor - User Manual Verification 'Phase 2' (Protocol in workflow.md)

---

## Phase 3: Live Order Placement

- [x] Task: Write failing tests for `place_market_order` (Red)
    - [x] Test that with `LIVE_TRADING` unset, function logs intent and returns `None`
    - [x] Test that with `LIVE_TRADING=true`, function calls `POST /api/v3/brokerage/orders`
    - [x] Test that a successful response with `success: true` returns the order object
    - [x] Test that a response with `success: false` raises `OrderError` with `error_response`
- [x] Task: Implement `place_market_order` in `coinbase_auth.py` (Green)
    - [x] Gate on `LIVE_TRADING=true` env var
    - [x] Build order payload: `client_order_id`, `product_id`, `side`, `order_configuration`
    - [x] POST to `/api/v3/brokerage/orders` via `CoinbaseClient`
    - [x] Validate and return response
- [x] Task: Conductor - User Manual Verification 'Phase 3' (Protocol in workflow.md)

---

## Phase 4: Integration with Existing Bot

- [x] Task: Write failing integration tests for updated `coinbase_live_trading.py` (Red)
    - [x] Test that bot initializes without error when credentials are present
    - [x] Test that simulation mode works without `LIVE_TRADING` set
    - [x] Test that bot uses `CoinbaseClient` (not old HMAC signing) for price fetching
- [x] Task: Update `coinbase_live_trading.py` to use `CoinbaseClient` (Green)
    - [x] Replace broken HMAC auth calls with `CoinbaseClient`
    - [x] Preserve simulation mode behaviour
    - [x] Remove dead auth code (`coinbase_auth_working.py`, `coinbase_auth_research.py` references)
- [x] Task: Run full test suite and verify >80% coverage on new modules
- [x] Task: Conductor - User Manual Verification 'Phase 4' (Protocol in workflow.md)
