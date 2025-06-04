package com.vsiwest.moneyfan.ingestion;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.vsiwest.moneyfan.coinbase.CoinbaseApiClient;
import com.vsiwest.moneyfan.coinbase.CoinbaseApiException; // Import custom exception
import com.vsiwest.moneyfan.config.CoinbaseApiConfig;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.Map;
// No change to imports needed other than CoinbaseApiException if not already present

public class CoinbaseSkimmer {

    private static final Logger logger = LoggerFactory.getLogger(CoinbaseSkimmer.class);
    private static final long FETCH_INTERVAL_MS = 60 * 1000; // 60 seconds

    public static void main(String[] args) {
        logger.info("Starting CoinbaseSkimmer...");

        CoinbaseApiConfig apiConfig;
        try {
            apiConfig = new CoinbaseApiConfig();
            logger.info("Coinbase API configuration loaded successfully.");
        } catch (IllegalStateException e) {
            logger.error("Failed to load Coinbase API configuration: {}", e.getMessage());
            logger.error("Please ensure COINBASE_API_KEY and COINBASE_API_SECRET environment variables are set.");
            logger.info("CoinbaseSkimmer exiting due to configuration error.");
            return;
        }

        CoinbaseApiClient apiClient = new CoinbaseApiClient(apiConfig);
        ObjectMapper objectMapper = new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT);

        logger.info("CoinbaseSkimmer started. Fetching balances every {} ms.", FETCH_INTERVAL_MS);

        try {
            while (true) {
                try {
                    logger.info("Fetching account balances...");
                    List<Map<String, Object>> balances = apiClient.getAccountBalances();
                    logger.info("Successfully fetched {} account(s).", balances.size());

                        // Pretty print to console, consider logging level for production
                        // For now, using INFO to ensure it's visible with default settings.
                        // If too verbose, this could be DEBUG or TRACE and rely on logger config.
                        System.out.println("Fetched balances at " + java.time.LocalDateTime.now() + ":");
                        System.out.println(objectMapper.writeValueAsString(balances));
                        logger.debug("Full balances JSON: {}", objectMapper.writeValueAsString(balances));


                } catch (CoinbaseApiException e) {
                    logger.error("Coinbase API error while fetching balances: {}", e.getMessage(), e);
                } catch (Exception e) { // Catch any other unexpected exceptions
                    logger.error("Unexpected error during skimming loop: {}", e.getMessage(), e);
                }

                logger.debug("Waiting for {} ms before next fetch.", FETCH_INTERVAL_MS);
                Thread.sleep(FETCH_INTERVAL_MS);
            }
        } catch (InterruptedException e) {
            logger.info("CoinbaseSkimmer interrupted. Exiting...");
            Thread.currentThread().interrupt();
        }
    }
}
