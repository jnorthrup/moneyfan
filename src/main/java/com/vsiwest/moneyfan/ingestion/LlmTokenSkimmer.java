package com.vsiwest.moneyfan.ingestion;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import com.vsiwest.moneyfan.client.LlmTokenApiClient;
import com.vsiwest.moneyfan.model.LlmToken;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

public class LlmTokenSkimmer {

    private static final Logger logger = LoggerFactory.getLogger(LlmTokenSkimmer.class);
    private static final long FETCH_INTERVAL_MS = 60 * 1000; // 60 seconds

    private final LlmTokenApiClient apiClient;
    private final ObjectMapper objectMapper;

    public LlmTokenSkimmer(LlmTokenApiClient apiClient) {
        this.apiClient = apiClient;
        this.objectMapper = new ObjectMapper()
                .registerModule(new JavaTimeModule()) // For Instant serialization
                .enable(SerializationFeature.INDENT_OUTPUT);
    }

    public void start() {
        logger.info("LlmTokenSkimmer started. Fetching LLM token data every {} ms.", FETCH_INTERVAL_MS);
        try {
            while (true) {
                try {
                    logger.info("Fetching LLM token data...");
                    List<LlmToken> tokens = apiClient.fetchLlmTokens();
                    logger.info("Successfully fetched {} LLM token(s).", tokens.size());

                    // Process or store the token data as needed
                    // For now, just printing to console
                    System.out.println("Fetched LLM Tokens at " + java.time.LocalDateTime.now() + ":");
                    System.out.println(objectMapper.writeValueAsString(tokens));
                    logger.debug("Full LLM Tokens JSON: {}", objectMapper.writeValueAsString(tokens));

                } catch (Exception e) { // Catch any unexpected exceptions during fetching or processing
                    logger.error("Unexpected error during LLM token skimming loop: {}", e.getMessage(), e);
                }

                logger.debug("Waiting for {} ms before next LLM token fetch.", FETCH_INTERVAL_MS);
                Thread.sleep(FETCH_INTERVAL_MS);
            }
        } catch (InterruptedException e) {
            logger.info("LlmTokenSkimmer interrupted. Exiting...");
            Thread.currentThread().interrupt();
        }
    }

    // Example main method for standalone execution (optional)
    public static void main(String[] args) {
        logger.info("Starting LlmTokenSkimmer (standalone)...");
        // In a real app, apiKey would come from a config
        LlmTokenApiClient client = new LlmTokenApiClient("dummy-api-key");
        LlmTokenSkimmer skimmer = new LlmTokenSkimmer(client);
        skimmer.start();
    }
}
