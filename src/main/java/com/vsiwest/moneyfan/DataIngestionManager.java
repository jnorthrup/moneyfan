package com.vsiwest.moneyfan;

import com.vsiwest.moneyfan.client.IntegrationApiClient;
import com.vsiwest.moneyfan.client.LlmTokenApiClient;
import com.vsiwest.moneyfan.client.ToolApiClient;
import com.vsiwest.moneyfan.config.CoinbaseApiConfig;
import com.vsiwest.moneyfan.coinbase.CoinbaseApiClient;
import com.vsiwest.moneyfan.ingestion.CoinbaseSkimmer;
import com.vsiwest.moneyfan.ingestion.IntegrationSkimmer;
import com.vsiwest.moneyfan.ingestion.LlmTokenSkimmer;
import com.vsiwest.moneyfan.ingestion.ToolSkimmer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
// import java.util.concurrent.TimeUnit; // Keep if shutdown hook is re-enabled

public class DataIngestionManager {

    private static final Logger logger = LoggerFactory.getLogger(DataIngestionManager.class);

    public static void main(String[] args) {
        logger.info("Starting Data Ingestion Manager...");

        ExecutorService executorService = Executors.newCachedThreadPool();

        // Initialize and submit CoinbaseSkimmer
        try {
            CoinbaseApiConfig coinbaseApiConfig = new CoinbaseApiConfig(); // Assumes env vars are set
            CoinbaseApiClient coinbaseApiClient = new CoinbaseApiClient(coinbaseApiConfig);
            CoinbaseSkimmer coinbaseSkimmer = new CoinbaseSkimmer(coinbaseApiClient); // Instantiate
            executorService.submit(coinbaseSkimmer::start); // Call instance method
            logger.info("CoinbaseSkimmer submitted to executor service.");
        } catch (IllegalStateException e) {
            logger.error("Failed to initialize or start CoinbaseSkimmer (config error): {}.", e.getMessage());
        } catch (Exception e) {
            logger.error("An unexpected error occurred during CoinbaseSkimmer setup: {}", e.getMessage(), e);
        }

        // Initialize and submit LlmTokenSkimmer
        try {
            LlmTokenApiClient llmTokenApiClient = new LlmTokenApiClient("dummy-llm-api-key");
            LlmTokenSkimmer llmTokenSkimmer = new LlmTokenSkimmer(llmTokenApiClient);
            executorService.submit(llmTokenSkimmer::start);
            logger.info("LlmTokenSkimmer submitted to executor service.");
        } catch (Exception e) {
            logger.error("Failed to start LlmTokenSkimmer: {}", e.getMessage(), e);
        }

        // Initialize and submit ToolSkimmer
        try {
            ToolApiClient toolApiClient = new ToolApiClient();
            ToolSkimmer toolSkimmer = new ToolSkimmer(toolApiClient);
            executorService.submit(toolSkimmer::start);
            logger.info("ToolSkimmer submitted to executor service.");
        } catch (Exception e) {
            logger.error("Failed to start ToolSkimmer: {}", e.getMessage(), e);
        }

        // Initialize and submit IntegrationSkimmer
        try {
            IntegrationApiClient integrationApiClient = new IntegrationApiClient();
            IntegrationSkimmer integrationSkimmer = new IntegrationSkimmer(integrationApiClient);
            executorService.submit(integrationSkimmer::start);
            logger.info("IntegrationSkimmer submitted to executor service.");
        } catch (Exception e) {
            logger.error("Failed to start IntegrationSkimmer: {}", e.getMessage(), e);
        }

        logger.info("All skimmers submitted to executor service. Manager will keep running.");

        // Optional: Add shutdown hook for graceful termination
        // Runtime.getRuntime().addShutdownHook(new Thread(() -> {
        //     logger.info("Shutdown hook triggered. Shutting down executor service...");
        //     executorService.shutdown();
        //     try {
        //         if (!executorService.awaitTermination(60, TimeUnit.SECONDS)) {
        //             executorService.shutdownNow();
        //         }
        //     } catch (InterruptedException ie) {
        //         executorService.shutdownNow();
        //         Thread.currentThread().interrupt();
        //     }
        //     logger.info("Executor service shut down.");
        // }));
    }
}
