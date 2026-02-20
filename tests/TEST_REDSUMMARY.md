# TDD Red Phase Summary - Coinbase JWT Authentication Tests

## Status: RED ✓
All 15 tests are failing as expected (ModuleNotFoundError: No module named 'coinbase_auth')

## Test File Created
**Location:** `/Users/jim/work/moneyfan/tests/test_coinbase_auth.py`
**Lines:** 345
**Test Classes:** 3
**Test Methods:** 15

## Test Coverage

### 1. TestCoinbaseAuthJWTGeneration (13 tests)
✓ `test_import_coinbase_auth_module` - Verifies module can be imported
✓ `test_generate_jwt_token_returns_valid_string` - JWT format validation
✓ `test_jwt_header_contains_es256_and_kid` - ES256 algorithm + kid in header
✓ `test_jwt_payload_contains_required_claims` - All required claims present
✓ `test_jwt_uri_claim_matches_request` - URI claim format validation
✓ `test_jwt_expiration_within_120_seconds` - exp = nbf + 120s
✓ `test_jwt_signature_is_valid_es256` - Signature validation
✓ `test_missing_api_key_name_raises_environment_error` - API key validation
✓ `test_missing_private_key_raises_environment_error` - Private key validation
✓ `test_empty_api_key_name_raises_environment_error` - Empty API key check
✓ `test_empty_private_key_raises_environment_error` - Empty private key check
✓ `test_different_methods_hosts_paths_generate_different_tokens` - Token uniqueness

### 2. TestCoinbaseAuthKeyFormats (1 test)
✓ `test_raw_ec_pem_format_supported` - Raw EC PEM format support

### 3. TestCoinbaseAuthModuleInterface (1 test)
✓ `test_generate_jwt_token_is_callable` - Function is callable
✓ `test_generate_jwt_token_accepts_method_host_path` - Parameter validation

## Acceptance Criteria Verification

✓ Test file `tests/test_coinbase_auth.py` created
✓ Tests cover valid JWT returned
✓ Tests cover ES256 alg + kid in header
✓ Tests cover correct claims (sub/iss/nbf/exp/uri)
✓ Tests cover exp within 120s of nbf
✓ Tests cover missing env vars raise EnvironmentError
✓ Tests FAIL when run (ImportError)

## Test Results
```
FAILED tests/test_coinbase_auth.py::TestCoinbaseAuthJWTGeneration::test_import_coinbase_auth_module
FAILED tests/test_coinbase_auth.py::TestCoinbaseAuthJWTGeneration::test_generate_jwt_token_returns_valid_string
... (15 total failures)
============================== 15 failed in 0.07s ==============================
```

## Next Steps (Green Phase)
Implement `coinbase_auth.py` module with:
- `generate_jwt_token(method, host, path)` function
- ES256 signature with EC private key
- JWT header: {alg: "ES256", kid: <key-id>}
- JWT payload: {sub, iss: "cdp", nbf, exp, uri}
- Environment variable loading
- Error handling for missing credentials
