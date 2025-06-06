package com.vsiwest.moneyfan.config;

import org.junit.jupiter.api.Test;
import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CoinbaseApiConfigTest {

    @Test
    void constructor_shouldLoadApiKeyAndSecret_whenEnvVariablesAreSet() {
        Map<String, String> envMap = new HashMap<>();
        envMap.put("COINBASE_API_KEY", "test_api_key");
        envMap.put("COINBASE_API_SECRET", "test_api_secret");
        CoinbaseApiConfig config = new CoinbaseApiConfig(envMap);
        assertEquals("test_api_key", config.getApiKey());
        assertEquals("test_api_secret", config.getApiSecret());
    }

    @Test
    void constructor_shouldThrowIllegalStateException_whenApiKeyIsMissing() {
        Map<String, String> envMap = new HashMap<>();
        envMap.put("COINBASE_API_SECRET", "test_api_secret");
        IllegalStateException exception = assertThrows(IllegalStateException.class, () -> new CoinbaseApiConfig(envMap));
        assertEquals("Missing environment variable: COINBASE_API_KEY", exception.getMessage());
    }

    @Test
    void constructor_shouldThrowIllegalStateException_whenApiSecretIsMissing() {
        Map<String, String> envMap = new HashMap<>();
        envMap.put("COINBASE_API_KEY", "test_api_key");
        IllegalStateException exception = assertThrows(IllegalStateException.class, () -> new CoinbaseApiConfig(envMap));
        assertEquals("Missing environment variable: COINBASE_API_SECRET", exception.getMessage());
    }
}
