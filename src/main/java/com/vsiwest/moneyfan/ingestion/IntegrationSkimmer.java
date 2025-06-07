package com.vsiwest.moneyfan.ingestion;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import com.vsiwest.moneyfan.client.IntegrationApiClient;
import com.vsiwest.moneyfan.model.Integration;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

public class IntegrationSkimmer {

    private static final Logger logger = LoggerFactory.getLogger(IntegrationSkimmer.class);
    private static final long FETCH_INTERVAL_MS = 60 * 1000 * 10; // 10 minutes, integrations change less frequently

    private final IntegrationApiClient apiClient;
    private final ObjectMapper objectMapper;

    public IntegrationSkimmer(IntegrationApiClient apiClient) {
        this.apiClient = apiClient;
        this.objectMapper = new ObjectMapper()
                .registerModule(new JavaTimeModule()) // For Instant serialization
                .enable(SerializationFeature.INDENT_OUTPUT);
    }

    public void start() {
        logger.info("IntegrationSkimmer started. Fetching integration data every {} ms.", FETCH_INTERVAL_MS);
        try {
            while (true) {
                try {
                    logger.info("Fetching integration data...");
                    List<Integration> integrations = apiClient.fetchIntegrations();
                    logger.info("Successfully fetched {} integration(s).", integrations.size());

                    // Process or store the integration data as needed
                    System.out.println("Fetched Integrations at " + java.time.LocalDateTime.now() + ":");
                    System.out.println(objectMapper.writeValueAsString(integrations));
                    logger.debug("Full Integrations JSON: {}", objectMapper.writeValueAsString(integrations));

                } catch (Exception e) {
                    logger.error("Unexpected error during integration skimming loop: {}", e.getMessage(), e);
                }

                logger.debug("Waiting for {} ms before next integration fetch.", FETCH_INTERVAL_MS);
                Thread.sleep(FETCH_INTERVAL_MS);
            }
        } catch (InterruptedException e) {
            logger.info("IntegrationSkimmer interrupted. Exiting...");
            Thread.currentThread().interrupt();
        }
    }

    // Example main method for standalone execution (optional)
    public static void main(String[] args) {
        logger.info("Starting IntegrationSkimmer (standalone)...");
        IntegrationApiClient client = new IntegrationApiClient();
        IntegrationSkimmer skimmer = new IntegrationSkimmer(client);
        skimmer.start();
    }
}
