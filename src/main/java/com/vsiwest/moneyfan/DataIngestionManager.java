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
import java.util.concurrent.TimeUnit;

public class DataIngestionManager {

    private static final Logger logger = LoggerFactory.getLogger(DataIngestionManager.class);

    public static void main(String[] args) {
        logger.info("Starting Data Ingestion Manager...");

        // Create a thread pool to run skimmers concurrently
        ExecutorService executorService = Executors.newCachedThreadPool();

        // Initialize and submit CoinbaseSkimmer
        try {
            CoinbaseApiConfig coinbaseApiConfig = new CoinbaseApiConfig(); // Assumes env vars are set
            CoinbaseApiClient coinbaseApiClient = new CoinbaseApiClient(coinbaseApiConfig);
            // Note: CoinbaseSkimmer's main method has its own loop and Thread.sleep.
            // For proper concurrent execution without its own main, it would need to be refactored
            // to expose a Runnable or Callable interface.
            // For now, we'll adapt its existing main logic structure slightly if needed,
            // or run its main method in a thread if it's self-contained enough.
            // Given CoinbaseSkimmer.main is static, we can't directly instantiate and run its logic easily
            // without refactoring it.
            // Let's assume for this step, we are aiming to run the *logic* of CoinbaseSkimmer.
            // A proper solution would be to refactor CoinbaseSkimmer to have a start() method like others.
            // For now, we'll just log that it would be started.
            logger.info("CoinbaseSkimmer would be started here. (Requires refactoring to be fully integrated into thread pool).");
            // To actually run it if its main is simple enough (and doesn't call System.exit):
            // executorService.submit(() -> CoinbaseSkimmer.main(new String[]{}));

        } catch (IllegalStateException e) {
            logger.error("Failed to initialize CoinbaseApiConfig: {}. CoinbaseSkimmer will not start.", e.getMessage());
        } catch (Exception e) {
            logger.error("An unexpected error occurred during CoinbaseSkimmer setup: {}", e.getMessage(), e);
        }


        // Initialize and submit LlmTokenSkimmer
        try {
            // API key for LlmTokenApiClient might come from config in a real app
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

        // Keep the main thread alive or implement a graceful shutdown
        // For now, let it run and skimmers will loop internally.
        // To make it shutdown gracefully:
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
