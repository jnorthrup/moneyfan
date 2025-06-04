package com.vsiwest.moneyfan.config;

public class CoinbaseApiConfig {

    private static final String COINBASE_API_KEY_ENV_VAR = "COINBASE_API_KEY";
    private static final String COINBASE_API_SECRET_ENV_VAR = "COINBASE_API_SECRET";

    private final String apiKey;
    private final String apiSecret;

    // Production constructor
    public CoinbaseApiConfig() {
        this(System.getenv());
    }

    // Test constructor
    public CoinbaseApiConfig(java.util.Map<String, String> envProvider) {
        this.apiKey = envProvider.get(COINBASE_API_KEY_ENV_VAR);
        this.apiSecret = envProvider.get(COINBASE_API_SECRET_ENV_VAR);

        if (this.apiKey == null || this.apiKey.isEmpty()) {
            throw new IllegalStateException("Missing environment variable: " + COINBASE_API_KEY_ENV_VAR);
        }

        if (this.apiSecret == null || this.apiSecret.isEmpty()) {
            throw new IllegalStateException("Missing environment variable: " + COINBASE_API_SECRET_ENV_VAR);
        }
    }

    public String getApiKey() {
        return apiKey;
    }

    public String getApiSecret() {
        return apiSecret;
    }
}
