package com.example.dsel.ingestion.service;

import com.example.dsel.ingestion.config.AppConfig;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.Collections;

public class IncrementalUpdateManager {

    private final AppConfig appConfig;
    private final String basePath;

    public IncrementalUpdateManager(AppConfig appConfig) {
        this.appConfig = appConfig;
        // Resolve home directory in base path
        String rawBasePath = appConfig.getMpImportBasePath();
        if (rawBasePath.startsWith("~" + java.io.File.separator) || rawBasePath.equals("~")) {
            this.basePath = System.getProperty("user.home") + rawBasePath.substring(1);
        } else if (rawBasePath.startsWith("~")) { 
            // Handle cases where there might not be a separator after ~ but it's not just ~
            // This is less common, usually it's ~/something or just ~
            // For safety, assuming if it starts with ~ and not ~/ then it's just ~
            this.basePath = System.getProperty("user.home");
        }
        else {
            this.basePath = rawBasePath;
        }
    }

    private Path getMetadataFilePath(String assetPair, String interval, String dataType) {
        // Sanitize assetPair for directory creation (e.g., BTC/USDT -> BTCUSDT)
        String sanitizedAssetPair = assetPair.replace("/", "");
        String filename = "last_" + dataType + "_timestamp.txt";
        // Example path: /path/to/mpdata/import/klines/1m/BTCUSDT/.meta/last_klines_timestamp.txt
        return Paths.get(basePath, dataType, interval, sanitizedAssetPair, ".meta", filename);
    }

    public long getLastTimestamp(String assetPair, String interval, String dataType) {
        Path metadataFile = getMetadataFilePath(assetPair, interval, dataType);
        if (Files.exists(metadataFile)) {
            try {
                String content = Files.readString(metadataFile);
                return Long.parseLong(content.trim());
            } catch (IOException | NumberFormatException e) {
                System.err.println("Error reading or parsing timestamp from " + metadataFile + ": " + e.getMessage());
                // Fallback to 0 if file is corrupted or unreadable
                return 0L;
            }
        }
        return 0L; // Default if no file exists (fetch from beginning)
    }

    public void updateLastTimestamp(String assetPair, String interval, String dataType, long newTimestamp) {
        Path metadataFile = getMetadataFilePath(assetPair, interval, dataType);
        try {
            Files.createDirectories(metadataFile.getParent()); // Ensure .meta directory exists
            Files.writeString(metadataFile, String.valueOf(newTimestamp), 
                              StandardOpenOption.CREATE, 
                              StandardOpenOption.WRITE, 
                              StandardOpenOption.TRUNCATE_EXISTING);
        } catch (IOException e) {
            System.err.println("Error writing timestamp to " + metadataFile + ": " + e.getMessage());
        }
    }
}
