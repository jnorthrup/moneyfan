package com.vsiwest.moneyfan.ingestion;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule; // Added for consistency if needed by ObjectMapper
import com.vsiwest.moneyfan.coinbase.CoinbaseApiClient;
import com.vsiwest.moneyfan.coinbase.CoinbaseApiException;
import com.vsiwest.moneyfan.config.CoinbaseApiConfig;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.Map;

public class CoinbaseSkimmer {

    private static final Logger logger = LoggerFactory.getLogger(CoinbaseSkimmer.class);
    private static final long DEFAULT_FETCH_INTERVAL_MS = 60 * 1000; // 60 seconds

    private final long fetchIntervalMs;
    private final CoinbaseApiClient apiClient;
    private final ObjectMapper objectMapper;

    public CoinbaseSkimmer(CoinbaseApiClient apiClient, long fetchIntervalMs) {
        this.apiClient = apiClient;
        this.fetchIntervalMs = fetchIntervalMs;
        this.objectMapper = new ObjectMapper()
                .registerModule(new JavaTimeModule()) // Ensure JavaTimeModule is registered
                .enable(SerializationFeature.INDENT_OUTPUT);
    }

    public CoinbaseSkimmer(CoinbaseApiClient apiClient) {
        this(apiClient, DEFAULT_FETCH_INTERVAL_MS);
    }

    public void start() {
        logger.info("CoinbaseSkimmer started. Fetching balances every {} ms.", fetchIntervalMs);

        try {
            while (true) {
                try {
                    logger.info("Fetching account balances from Coinbase...");
                    List<Map<String, Object>> balances = apiClient.getAccountBalances();
                    logger.info("Successfully fetched {} account(s) from Coinbase.", balances.size());

                    // Pretty print to console or process further
                    System.out.println("Fetched Coinbase balances at " + java.time.LocalDateTime.now() + ":");
                    System.out.println(objectMapper.writeValueAsString(balances));
                    logger.debug("Full Coinbase balances JSON: {}", objectMapper.writeValueAsString(balances));

                    // Here, you could convert these Maps to CryptoAsset objects if desired:
                    // List<CryptoAsset> cryptoAssets = balances.stream()
                    //    .map(accountMap -> CryptoAsset.fromCoinbaseAccountMap(accountMap, "Coinbase"))
                    //    .collect(Collectors.toList());
                    // logger.info("Converted to {} CryptoAsset objects.", cryptoAssets.size());
                    // Then pass cryptoAssets to a generic data handler or the ArbitrageEngine

                } catch (CoinbaseApiException e) {
                    logger.error("Coinbase API error while fetching balances: {}", e.getMessage(), e);
                } catch (Exception e) { // Catch any other unexpected exceptions
                    logger.error("Unexpected error during Coinbase skimming loop: {}", e.getMessage(), e);
                }

                logger.debug("Waiting for {} ms before next Coinbase fetch.", fetchIntervalMs);
                Thread.sleep(fetchIntervalMs);
            }
        } catch (InterruptedException e) {
            logger.info("CoinbaseSkimmer interrupted. Exiting...");
            Thread.currentThread().interrupt();
        }
    }

    // Original main method can be kept for standalone execution or testing
    public static void main(String[] args) {
        logger.info("Starting CoinbaseSkimmer (standalone)...");

        CoinbaseApiConfig apiConfig;
        try {
            apiConfig = new CoinbaseApiConfig(); // Assumes env vars COINBASE_API_KEY and COINBASE_API_SECRET are set
            logger.info("Coinbase API configuration loaded successfully for standalone execution.");
        } catch (IllegalStateException e) {
            logger.error("Failed to load Coinbase API configuration for standalone execution: {}", e.getMessage());
            logger.error("Please ensure COINBASE_API_KEY and COINBASE_API_SECRET environment variables are set.");
            logger.info("CoinbaseSkimmer (standalone) exiting due to configuration error.");
            return;
        }

        CoinbaseApiClient apiClientInstance = new CoinbaseApiClient(apiConfig);
        CoinbaseSkimmer skimmer = new CoinbaseSkimmer(apiClientInstance);
        skimmer.start(); // Call the instance method
    }
}
