package com.vsiwest.moneyfan.client;

import com.vsiwest.moneyfan.model.Tool;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.math.BigDecimal;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;

// Placeholder for an API client that fetches Tool data
public class ToolApiClient {

    private static final Logger logger = LoggerFactory.getLogger(ToolApiClient.class);

    public ToolApiClient() {
        // Initialize HttpClient or other necessary components here
        logger.info("ToolApiClient initialized.");
    }

    // Placeholder method to fetch tool data
    // In a real implementation, this might involve web scraping or calling various vendor APIs
    public List<Tool> fetchTools() {
        logger.warn("fetchTools() is a placeholder and does not call a real API/scraper yet.");
        List<Tool> tools = new ArrayList<>();
        // Example placeholder data
        tools.add(new Tool("github-copilot", "GitHub Copilot", "Code Assistant", new BigDecimal("10.00"), "monthly", "GitHubWebsite", Instant.now()));
        tools.add(new Tool("jasper-ai-pro", "Jasper AI Pro", "Content Generation", new BigDecimal("59.00"), "monthly", "JasperWebsite", Instant.now()));
        return tools;
    }
}
