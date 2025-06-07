package com.vsiwest.moneyfan.ingestion;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import com.vsiwest.moneyfan.client.ToolApiClient;
import com.vsiwest.moneyfan.model.Tool;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

public class ToolSkimmer {

    private static final Logger logger = LoggerFactory.getLogger(ToolSkimmer.class);
    private static final long FETCH_INTERVAL_MS = 60 * 1000 * 5; // 5 minutes, tools change less frequently

    private final ToolApiClient apiClient;
    private final ObjectMapper objectMapper;

    public ToolSkimmer(ToolApiClient apiClient) {
        this.apiClient = apiClient;
        this.objectMapper = new ObjectMapper()
                .registerModule(new JavaTimeModule()) // For Instant serialization
                .enable(SerializationFeature.INDENT_OUTPUT);
    }

    public void start() {
        logger.info("ToolSkimmer started. Fetching tool data every {} ms.", FETCH_INTERVAL_MS);
        try {
            while (true) {
                try {
                    logger.info("Fetching tool data...");
                    List<Tool> tools = apiClient.fetchTools();
                    logger.info("Successfully fetched {} tool(s).", tools.size());

                    // Process or store the tool data as needed
                    System.out.println("Fetched Tools at " + java.time.LocalDateTime.now() + ":");
                    System.out.println(objectMapper.writeValueAsString(tools));
                    logger.debug("Full Tools JSON: {}", objectMapper.writeValueAsString(tools));

                } catch (Exception e) {
                    logger.error("Unexpected error during tool skimming loop: {}", e.getMessage(), e);
                }

                logger.debug("Waiting for {} ms before next tool fetch.", FETCH_INTERVAL_MS);
                Thread.sleep(FETCH_INTERVAL_MS);
            }
        } catch (InterruptedException e) {
            logger.info("ToolSkimmer interrupted. Exiting...");
            Thread.currentThread().interrupt();
        }
    }

    // Example main method for standalone execution (optional)
    public static void main(String[] args) {
        logger.info("Starting ToolSkimmer (standalone)...");
        ToolApiClient client = new ToolApiClient();
        ToolSkimmer skimmer = new ToolSkimmer(client);
        skimmer.start();
    }
}
