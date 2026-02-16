# Spec: Fix Coinbase Advanced Trade Authentication and Enable Live Order Placement

## Problem Statement

The existing Coinbase bot (`coinbase_bot_working.py`, `coinbase_live_trading.py`, etc.) fails with HTTP 401 Unauthorized when calling authenticated endpoints. The root cause is incorrect request signing: the codebase attempts HMAC-SHA256 signing against an EC private key, which is incompatible with the Coinbase Advanced Trade API v3 authentication scheme.

Coinbase Advanced Trade API v3 requires **JWT (JSON Web Token)** authentication signed with the EC private key using ES256 (ECDSA with SHA-256).

## Requirements

### R1: JWT Authentication Module
- Implement a reusable Python module (`coinbase_auth.py`) that generates valid JWT tokens for Coinbase Advanced Trade API v3.
- The JWT must be signed with the EC private key using the ES256 algorithm.
- JWT claims must include: `sub` (API key name), `iss` ("cdp"), `nbf` (now), `exp` (now + 120s), `uri` (method + host + path).
- The module must load credentials from environment variables (`COINBASE_API_KEY_NAME`, `COINBASE_PRIVATE_KEY`) and/or a `.env` file.
- Private key must support both raw EC PEM format and the CDP SDK key format.

### R2: Authenticated HTTP Client
- Implement a thin `CoinbaseClient` class that wraps `requests` (or `httpx`) and automatically injects the JWT `Authorization: Bearer <token>` header on every request.
- Must support GET and POST methods.
- Must handle 401 responses with a clear error message (do not silently retry).

### R3: Verified Authenticated Endpoints
The following endpoints must return 200 OK with valid data using the new auth:
- `GET /api/v3/brokerage/accounts` — list accounts and balances
- `GET /api/v3/brokerage/products/{product_id}` — get product info
- `GET /api/v3/brokerage/orders/historical/batch` — list order history

### R4: Live Order Placement (Simulation-Gated)
- Implement `place_market_order(product_id, side, base_size)` using `POST /api/v3/brokerage/orders`.
- The function must be gated behind a `LIVE_TRADING=true` environment variable. If not set, it must log the order intent without executing.
- Order response must be validated: check `success` field and surface any `error_response`.

### R5: Integration with Existing Bot
- Update `coinbase_live_trading.py` to use the new `CoinbaseClient` instead of the broken auth methods.
- Preserve existing simulation mode — live auth is only activated when `LIVE_TRADING=true`.

### R6: Tests
- Unit tests for JWT generation: verify token structure, claims, and signature validity using the `python-jwt` or `PyJWT` library.
- Unit tests for `CoinbaseClient`: mock HTTP responses, verify `Authorization` header is set correctly.
- Integration smoke test (skipped in CI if no credentials): call `/api/v3/brokerage/accounts` with real credentials and assert HTTP 200.

## Acceptance Criteria
- `COINBASE_API_KEY_NAME` + `COINBASE_PRIVATE_KEY` env vars → successful JWT generation
- Calling `/api/v3/brokerage/accounts` returns HTTP 200 and a non-empty account list
- `place_market_order(...)` with `LIVE_TRADING` unset logs intent and does NOT call the API
- All unit tests pass with >80% coverage on new modules
- No 401 errors in authenticated calls

## Out of Scope
- Migrating the Kotlin/XChange bot
- WebSocket authenticated feeds
- Order cancellation or amendment
