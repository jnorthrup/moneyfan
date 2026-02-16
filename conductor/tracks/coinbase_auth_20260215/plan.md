# Plan: Fix Coinbase Advanced Trade Authentication and Enable Live Order Placement

Track ID: `coinbase_auth_20260215`

---

## Phase 1: JWT Authentication Module

- [x] Task: Write failing tests for JWT token generation (Red) [aac51f7]
    - [ ] Test that `generate_jwt(method, path)` returns a valid JWT string
    - [ ] Test that JWT header contains `alg: ES256` and `kid: <api_key_name>`
    - [ ] Test that JWT claims contain `sub`, `iss`, `nbf`, `exp`, `uri`
    - [ ] Test that `exp` is within 120 seconds of `nbf`
    - [ ] Test that missing env vars raise a clear `EnvironmentError`
- [x] Task: Implement `coinbase_auth.py` — JWT generation module (Green) [98833ad]
    - [ ] Load `COINBASE_API_KEY_NAME` and `COINBASE_PRIVATE_KEY` from env / `.env`
    - [ ] Parse EC private key (support raw PEM and CDP SDK format)
    - [ ] Sign JWT with ES256 using `PyJWT` or `python-jose`
    - [ ] Expose `generate_jwt(method: str, path: str) -> str`
- [ ] Task: Conductor - User Manual Verification 'Phase 1' (Protocol in workflow.md)

---

## Phase 2: Authenticated HTTP Client

- [ ] Task: Write failing tests for `CoinbaseClient` (Red)
    - [ ] Test that GET request includes `Authorization: Bearer <token>` header
    - [ ] Test that POST request includes `Authorization: Bearer <token>` header
    - [ ] Test that HTTP 401 response raises `AuthenticationError` with clear message
    - [ ] Test that client reads base URL from env (`COINBASE_API_URL`, default `https://api.coinbase.com`)
- [ ] Task: Implement `CoinbaseClient` class (Green)
    - [ ] Wrap `requests.Session` with auto JWT injection
    - [ ] Implement `get(path, **kwargs)` and `post(path, json, **kwargs)`
    - [ ] Handle 401 with descriptive error
- [ ] Task: Verify authenticated endpoints return 200 OK
    - [ ] `GET /api/v3/brokerage/accounts` → 200 + non-empty account list
    - [ ] `GET /api/v3/brokerage/products/BTC-USD` → 200 + product data
    - [ ] `GET /api/v3/brokerage/orders/historical/batch` → 200
- [ ] Task: Conductor - User Manual Verification 'Phase 2' (Protocol in workflow.md)

---

## Phase 3: Live Order Placement

- [ ] Task: Write failing tests for `place_market_order` (Red)
    - [ ] Test that with `LIVE_TRADING` unset, function logs intent and returns `None`
    - [ ] Test that with `LIVE_TRADING=true`, function calls `POST /api/v3/brokerage/orders`
    - [ ] Test that a successful response with `success: true` returns the order object
    - [ ] Test that a response with `success: false` raises `OrderError` with `error_response`
- [ ] Task: Implement `place_market_order` in `coinbase_auth.py` (Green)
    - [ ] Gate on `LIVE_TRADING=true` env var
    - [ ] Build order payload: `client_order_id`, `product_id`, `side`, `order_configuration`
    - [ ] POST to `/api/v3/brokerage/orders` via `CoinbaseClient`
    - [ ] Validate and return response
- [ ] Task: Conductor - User Manual Verification 'Phase 3' (Protocol in workflow.md)

---

## Phase 4: Integration with Existing Bot

- [ ] Task: Write failing integration tests for updated `coinbase_live_trading.py` (Red)
    - [ ] Test that bot initializes without error when credentials are present
    - [ ] Test that simulation mode works without `LIVE_TRADING` set
    - [ ] Test that bot uses `CoinbaseClient` (not old HMAC signing) for price fetching
- [ ] Task: Update `coinbase_live_trading.py` to use `CoinbaseClient` (Green)
    - [ ] Replace broken HMAC auth calls with `CoinbaseClient`
    - [ ] Preserve simulation mode behaviour
    - [ ] Remove dead auth code (`coinbase_auth_working.py`, `coinbase_auth_research.py` references)
- [ ] Task: Run full test suite and verify >80% coverage on new modules
- [ ] Task: Conductor - User Manual Verification 'Phase 4' (Protocol in workflow.md)
