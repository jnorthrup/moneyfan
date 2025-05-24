package com.example.dsel.ingestion.config;

import java.util.List;
import java.util.Arrays;

public class AppConfig {

    private final List<String> trackedAssets;
    private final List<String> targetTimeUnits;
    private final String mpImportBasePath;
    private final String mpCacheBasePath;
    private final String binanceApiKey;
    private final String binanceApiSecret;

    public AppConfig() {
        // Ideally, these configurations would be loaded from a dedicated configuration management system,
        // environment variables, or files (e.g., as indicated in the original Help.kt or project docs).

        this.trackedAssets = Arrays.asList(
            "ADAUP/USDT", "ADADOWN/USDT", "ADA/USDT", "BNBUP/USDT", "BNBDOWN/USDT", "BNB/USDT", 
            "BTCUP/USDT", "BTCDOWN/USDT", "BTC/USDT", "DOTUP/USDT", "DOTDOWN/USDT", "DOT/USDT", 
            "EOSUP/USDT", "EOSDOWN/USDT", "EOS/USDT", "ETHUP/USDT", "ETHDOWN/USDT", "ETH/USDT", 
            "LINKUP/USDT", "LINKDOWN/USDT", "LINK/USDT", "LTCUP/USDT", "LTCDOWN/USDT", "LTC/USDT", 
            "SUSHIUP/USDT", "SUSHIDOWN/USDT", "SUSHI/USDT", "TRXUP/USDT", "TRXDOWN/USDT", "TRX/USDT", 
            "XLMUP/USDT", "XLMDOWN/USDT", "XLM/USDT", "XTZUP/USDT", "XTZDOWN/USDT", "XTZ/USDT"
        );

        this.targetTimeUnits = Arrays.asList("1m", "1h", "1d");
        
        // The '~' character in these paths typically represents the user's home directory.
        // This path resolution would need to be handled by the application at runtime.
        this.mpImportBasePath = "~/mpdata/import"; 
        this.mpCacheBasePath = "~/mpdata/cache"; 
        
        // API keys should be loaded securely, for example, from environment variables
        // or a protected file (e.g., ~/mpdata/.keys/testnet.csv as mentioned in user context).
        // Using placeholder values for now.
        this.binanceApiKey = "YOUR_API_KEY_HERE"; 
        this.binanceApiSecret = "YOUR_API_SECRET_HERE";
    }

    public List<String> getTrackedAssets() {
        return trackedAssets;
    }

    public List<String> getTargetTimeUnits() {
        return targetTimeUnits;
    }

    public String getMpImportBasePath() {
        // Note: Path resolution for '~' is not handled here.
        // The component using this path is responsible for resolving it.
        return mpImportBasePath;
    }

    public String getMpCacheBasePath() {
        // Note: Path resolution for '~' is not handled here.
        return mpCacheBasePath;
    }

    public String getBinanceApiKey() {
        return binanceApiKey;
    }

    public String getBinanceApiSecret() {
        return binanceApiSecret;
    }
}
