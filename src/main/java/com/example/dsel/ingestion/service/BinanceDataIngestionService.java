package com.example.dsel.ingestion.service;

import com.example.dsel.ingestion.config.AppConfig;
import com.example.dsel.ingestion.dto.BinanceKline;
import com.example.dsel.ingestion.dto.BinanceTrade;
import com.example.dsel.ingestion.schema.DselSchemas;
import borg.trikeshed.cursor.Cursor;
// import borg.trikeshed.lib.Series; // No longer directly used here, Cursor.size() is available

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Instant;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.time.YearMonth;
import java.time.temporal.ChronoUnit;
import java.util.ArrayList;
import java.util.List;
import java.util.TimeZone;
import java.util.concurrent.TimeUnit;
import java.math.BigDecimal; // For DTO parsing
import com.example.dsel.ingestion.util.IndexedCsvReader; // Added import

public class BinanceDataIngestionService {

    private static final String KLINE_HEADER = "Open_time,Open,High,Low,Close,Volume,Close_time,Quote_asset_volume,Number_of_trades,Taker_buy_base_asset_volume,Taker_buy_quote_asset_volume,Ignore";
    private static final String TRADE_HEADER = "trade_Id,price,qty,quoteQty,time,isBuyerMaker,isBestMatch";

    private final AppConfig appConfig;
    private final ExternalDataFetcher externalDataFetcher; // Added
    private final DataTransformerService dataTransformerService;
    private final IncrementalUpdateManager incrementalUpdateManager;
    private final IsamPersistenceService isamPersistenceService;
    // Removed BinanceApiWrapper as direct API calls for recency gap are now placeholder

    public BinanceDataIngestionService(AppConfig appConfig,
                                       ExternalDataFetcher externalDataFetcher, // Added
                                       DataTransformerService dataTransformerService,
                                       IncrementalUpdateManager incrementalUpdateManager,
                                       IsamPersistenceService isamPersistenceService) {
        this.appConfig = appConfig;
        this.externalDataFetcher = externalDataFetcher;
        this.dataTransformerService = dataTransformerService;
        this.incrementalUpdateManager = incrementalUpdateManager;
        this.isamPersistenceService = isamPersistenceService;
    }

    public void ingestAllData() {
        List<String> assets = appConfig.getTrackedAssets();
        List<String> intervals = appConfig.getTargetTimeUnits(); // e.g., "1m", "1h", "1d"

        for (String assetPair : assets) {
            for (String interval : intervals) {
                orchestrateDataIngestion(assetPair, interval, "klines");
                orchestrateDataIngestion(assetPair, interval, "trades");
            }
        }
        System.out.println("Data ingestion cycle complete.");
    }

    private void orchestrateDataIngestion(String assetPair, String interval, String dataType) {
        System.out.printf("Orchestrating ingestion for %s - %s - %s%n", assetPair, interval, dataType);
        String sanitizedAssetPair = assetPair.replace("/", "");
        Path mpImportDir = Paths.get(resolvePath(appConfig.getMpImportBasePath()), dataType, interval, sanitizedAssetPair);
        try {
            Files.createDirectories(mpImportDir); // Ensure directory exists
        } catch (IOException e) {
            System.err.printf("Could not create directory %s: %s%n", mpImportDir, e.getMessage());
            return;
        }
        Path mainCsvFile = mpImportDir.resolve("final-" + sanitizedAssetPair + "-" + interval + ".csv");
        Path tempDirForMerge = Paths.get(resolvePath(appConfig.getMpCacheBasePath()), "temp_merge"); // For merge tool

        try {
            Files.createDirectories(tempDirForMerge);

            long lastProcessedTimestamp = incrementalUpdateManager.getLastTimestamp(assetPair, interval, dataType);
            System.out.printf("Last processed timestamp for %s %s %s: %d%n", assetPair, interval, dataType, lastProcessedTimestamp);

            // 1. Initial Bulk Sync (if main CSV doesn't exist or last timestamp is very old)
            if (!Files.exists(mainCsvFile) || lastProcessedTimestamp == 0L) {
                System.out.printf("Performing initial bulk sync for %s %s %s to %s%n", assetPair, interval, dataType, mainCsvFile);
                Path bulkCsv;
                if (dataType.equals("klines")) {
                    bulkCsv = externalDataFetcher.fetchAndProcessKlines(assetPair, interval); // This should create the initial mainCsvFile
                } else {
                    bulkCsv = externalDataFetcher.fetchAndProcessTrades(assetPair, interval);
                }
                if (bulkCsv != null && Files.exists(bulkCsv)) {
                     if (!mainCsvFile.equals(bulkCsv)) { // If fetcher saves to a different name initially
                        Files.move(bulkCsv, mainCsvFile, java.nio.file.StandardCopyOption.REPLACE_EXISTING);
                     }
                    lastProcessedTimestamp = getLastTimestampFromCsv(mainCsvFile, dataType);
                    System.out.printf("Initial bulk sync complete. New last timestamp from CSV: %d%n", lastProcessedTimestamp);
                } else {
                    System.out.printf("Initial bulk sync failed or produced no data for %s %s %s.%n", assetPair, interval, dataType);
                }
            }

            // 2. Iterative Gap Filling (Archives)
            LocalDate today = LocalDate.now(ZoneOffset.UTC);
            // Stop 2 days before today to allow archives to publish
            LocalDate fillUntilDate = today.minusDays(2); 
            
            LocalDateTime lastTimestampDateTime = LocalDateTime.ofInstant(Instant.ofEpochMilli(lastProcessedTimestamp), ZoneOffset.UTC);

            while (lastTimestampDateTime.toLocalDate().isBefore(fillUntilDate)) {
                Path periodDataCsv = null;
                LocalDate nextFetchDate = lastTimestampDateTime.toLocalDate().plusDays(1); // Start with day after last data

                if (dataType.equals("klines")) { // Klines can be fetched daily or monthly
                    // Prioritize monthly if a full month gap, then daily
                    if (nextFetchDate.getDayOfMonth() == 1 && YearMonth.from(nextFetchDate).isBefore(YearMonth.from(fillUntilDate))) {
                        System.out.printf("Attempting monthly kline fetch for %s %s for %d-%02d%n", assetPair, interval, nextFetchDate.getYear(), nextFetchDate.getMonthValue());
                        periodDataCsv = externalDataFetcher.fetchAndProcessSpecificMonthlyKlines(assetPair, interval, nextFetchDate.getYear(), nextFetchDate.getMonthValue());
                        if (periodDataCsv != null && Files.exists(periodDataCsv)) {
                           lastTimestampDateTime = lastTimestampDateTime.plusMonths(1).withDayOfMonth(1); // Assume full month fetched
                        } else { // If monthly failed or no data, try daily for that first day of month
                             System.out.printf("Monthly kline fetch failed or no data, trying daily for %s %s for %s%n", assetPair, interval, nextFetchDate);
                             periodDataCsv = externalDataFetcher.fetchAndProcessSpecificDailyKlines(assetPair, interval, nextFetchDate);
                             if (periodDataCsv != null && Files.exists(periodDataCsv)){
                                lastTimestampDateTime = lastTimestampDateTime.plusDays(1);
                             } else {
                                 System.out.printf("No data for daily kline %s after monthly attempt. Breaking gap fill.%n", nextFetchDate);
                                 break; 
                             }
                        }
                    } else {
                        System.out.printf("Attempting daily kline fetch for %s %s for %s%n", assetPair, interval, nextFetchDate);
                        periodDataCsv = externalDataFetcher.fetchAndProcessSpecificDailyKlines(assetPair, interval, nextFetchDate);
                         if (periodDataCsv != null && Files.exists(periodDataCsv)){
                            lastTimestampDateTime = lastTimestampDateTime.plusDays(1);
                         } else {
                             System.out.printf("No data for daily kline %s. Breaking gap fill.%n", nextFetchDate);
                             break;
                         }
                    }
                } else { // Trades are fetched daily or monthly based on similar logic
                    if (nextFetchDate.getDayOfMonth() == 1 && YearMonth.from(nextFetchDate).isBefore(YearMonth.from(fillUntilDate))) {
                        System.out.printf("Attempting monthly trade fetch for %s %s for %d-%02d%n", assetPair, interval, nextFetchDate.getYear(), nextFetchDate.getMonthValue());
                        periodDataCsv = externalDataFetcher.fetchAndProcessSpecificMonthlyTrades(assetPair, interval, nextFetchDate.getYear(), nextFetchDate.getMonthValue());
                         if (periodDataCsv != null && Files.exists(periodDataCsv)) {
                           lastTimestampDateTime = lastTimestampDateTime.plusMonths(1).withDayOfMonth(1);
                        } else {
                             System.out.printf("Monthly trade fetch failed or no data, trying daily for %s %s for %s%n", assetPair, interval, nextFetchDate);
                             periodDataCsv = externalDataFetcher.fetchAndProcessSpecificDailyTrades(assetPair, interval, nextFetchDate);
                             if (periodDataCsv != null && Files.exists(periodDataCsv)){
                                lastTimestampDateTime = lastTimestampDateTime.plusDays(1);
                             } else {
                                 System.out.printf("No data for daily trade %s after monthly attempt. Breaking gap fill.%n", nextFetchDate);
                                 break;
                             }
                        }
                    } else {
                         System.out.printf("Attempting daily trade fetch for %s %s for %s%n", assetPair, interval, nextFetchDate);
                        periodDataCsv = externalDataFetcher.fetchAndProcessSpecificDailyTrades(assetPair, interval, nextFetchDate);
                        if (periodDataCsv != null && Files.exists(periodDataCsv)){
                            lastTimestampDateTime = lastTimestampDateTime.plusDays(1);
                         } else {
                             System.out.printf("No data for daily trade %s. Breaking gap fill.%n", nextFetchDate);
                             break;
                         }
                    }
                }

                if (periodDataCsv != null && Files.exists(periodDataCsv)) {
                    System.out.printf("Merging %s into %s%n", periodDataCsv, mainCsvFile);
                    externalDataFetcher.mergeAndSortCsvFiles(mainCsvFile, periodDataCsv, (dataType.equals("klines") ? KLINE_HEADER : TRADE_HEADER), tempDirForMerge);
                    // Update lastProcessedTimestamp from the merged main CSV
                    long newCsvTimestamp = getLastTimestampFromCsv(mainCsvFile, dataType);
                    if (newCsvTimestamp > lastProcessedTimestamp) {
                        lastProcessedTimestamp = newCsvTimestamp;
                        lastTimestampDateTime = LocalDateTime.ofInstant(Instant.ofEpochMilli(lastProcessedTimestamp), ZoneOffset.UTC); // Update for loop condition
                         System.out.printf("Gap fill successful for %s. New last timestamp from CSV: %d%n", nextFetchDate, lastProcessedTimestamp);
                    } else {
                        System.out.printf("Merge for %s did not advance timestamp. Current: %d. Breaking gap fill.%n", nextFetchDate, lastProcessedTimestamp);
                        Files.deleteIfExists(periodDataCsv); // Clean up processed daily/monthly file
                        break; 
                    }
                    Files.deleteIfExists(periodDataCsv); // Clean up processed daily/monthly file
                } else {
                    System.out.printf("No new data fetched for %s %s %s for %s. Gap filling might be complete or data unavailable.%n", assetPair, interval, dataType, nextFetchDate);
                    // If no data for a specific day/month, advance check to next period to avoid infinite loop
                    // This logic is implicitly handled by how lastTimestampDateTime is advanced above.
                    // If periodDataCsv is null, lastTimestampDateTime isn't advanced based on it, and the loop continues to the next iteration.
                }
            }

            // 3. Small Recency Gap Filling (Placeholder)
            System.out.println("// TODO: Implement direct API call for very recent gaps if needed for " + assetPair + " " + interval + " " + dataType);

            // 4. Final ISAM Conversion
            if (Files.exists(mainCsvFile) && Files.size(mainCsvFile) > 0) {
                System.out.printf("Starting ISAM conversion for %s%n", mainCsvFile);
                List<?> dtos; // List<BinanceKline> or List<BinanceTrade>
                if (dataType.equals("klines")) {
                    dtos = parseKlinesFromCsv(mainCsvFile);
                    if (!dtos.isEmpty()) {
                        Cursor klineCursor = dataTransformerService.transformKlines((List<BinanceKline>) dtos, DselSchemas.KLINE_SCHEMA);
                        if (klineCursor != null && klineCursor.size() > 0) {
                            isamPersistenceService.saveCursor(klineCursor, assetPair, interval, dataType, DselSchemas.KLINE_SCHEMA);
                            long finalTimestamp = ((BinanceKline) dtos.get(dtos.size() - 1)).closeTime();
                            incrementalUpdateManager.updateLastTimestamp(assetPair, interval, dataType, finalTimestamp);
                            System.out.printf("ISAM conversion complete for klines. Final timestamp: %d%n", finalTimestamp);
                        }
                    }
                } else { // trades
                    dtos = parseTradesFromCsv(mainCsvFile);
                     if (!dtos.isEmpty()) {
                        Cursor tradeCursor = dataTransformerService.transformTrades((List<BinanceTrade>) dtos, DselSchemas.TRADE_SCHEMA);
                        if (tradeCursor != null && tradeCursor.size() > 0) {
                            isamPersistenceService.saveCursor(tradeCursor, assetPair, interval, dataType, DselSchemas.TRADE_SCHEMA);
                            long finalTimestamp = ((BinanceTrade) dtos.get(dtos.size() - 1)).time();
                            incrementalUpdateManager.updateLastTimestamp(assetPair, interval, dataType, finalTimestamp);
                             System.out.printf("ISAM conversion complete for trades. Final timestamp: %d%n", finalTimestamp);
                        }
                    }
                }
                 if (dtos.isEmpty()) {
                    System.out.printf("No DTOs parsed from CSV %s. ISAM conversion skipped.%n", mainCsvFile);
                }
            } else {
                 System.out.printf("Main CSV %s is empty or does not exist. ISAM conversion skipped.%n", mainCsvFile);
            }

        } catch (Exception e) {
            System.err.printf("Error during data orchestration for %s - %s - %s: %s%n", assetPair, interval, dataType, e.getMessage());
            e.printStackTrace();
        } finally {
            try {
                // Clean up temp merge directory
                if (Files.exists(tempDirForMerge)) {
                    Files.walk(tempDirForMerge)
                         .sorted(java.util.Comparator.reverseOrder())
                         .map(Path::toFile)
                         .forEach(java.io.File::delete);
                }
            } catch (IOException ex) {
                System.err.println("Error cleaning up temp merge directory: " + ex.getMessage());
            }
        }
    }
    
    private String resolvePath(String pathStr) { // Helper from ExternalDataFetcher
        if (pathStr == null) return "."; 
        if (pathStr.startsWith("~" + java.io.File.separator) || pathStr.equals("~")) {
            return System.getProperty("user.home") + pathStr.substring(1);
        } else if (pathStr.startsWith("~")) {
            return System.getProperty("user.home");
        }
        return pathStr;
    }

    private long getLastTimestampFromCsv(Path csvFile, String dataType) throws IOException {
        if (!Files.exists(csvFile) || Files.size(csvFile) == 0) return 0L;
        
        // Using ProcessBuilder to replicate: tail -n 1 csvFile | cut -f<timestamp_column> -d,
        // This is simpler than reading potentially huge CSVs in Java just for the last line.
        // Kline: Open_time (col 1), Close_time (col 7). Use Close_time.
        // Trade: time (col 5)
        String timestampColumn = dataType.equals("klines") ? "7" : "5";
        ProcessBuilder pb = new ProcessBuilder("bash", "-c", 
            String.format("tail -n 1 %s | cut -f%s -d,", csvFile.toAbsolutePath().toString(), timestampColumn));
        Process process = pb.start();
        
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
            String lastLineTimestampStr = reader.readLine();
            if (lastLineTimestampStr != null && !lastLineTimestampStr.isEmpty()) {
                // Check if it's the header (e.g. if only header exists or error)
                if (lastLineTimestampStr.equalsIgnoreCase(dataType.equals("klines") ? "Close_time" : "time")) return 0L;
                return Long.parseLong(lastLineTimestampStr.trim());
            }
        } catch (NumberFormatException e) {
            System.err.println("Could not parse timestamp from last line of CSV " + csvFile + ": " + e.getMessage());
        } finally {
            try {
                process.waitFor(5, TimeUnit.SECONDS); // Wait a bit for process to avoid resource leak
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            if (process.isAlive()) process.destroyForcibly();
        }
        return 0L;
    }

    private List<BinanceKline> parseKlinesFromCsv(Path csvFile) throws IOException {
        List<BinanceKline> klines = new ArrayList<>();
        if (!Files.exists(csvFile) || Files.size(csvFile) == 0) {
            System.out.println("CSV file for klines is empty or does not exist: " + csvFile);
            return klines;
        }

        try (IndexedCsvReader reader = new IndexedCsvReader(csvFile)) {
            // String header = reader.getHeader(); // Optional: use header if needed for validation
            for (String line : reader) { // Iterate using the Iterable interface
                if (line.trim().isEmpty()) continue;
                String[] parts = line.split(",");
                if (parts.length < 12) {
                    System.err.println("Skipping kline CSV line due to insufficient parts: " + line);
                    continue;
                }
                try {
                    klines.add(new BinanceKline(
                        Long.parseLong(parts[0]), // openTime
                        parts[1],                 // open
                        parts[2],                 // high
                        parts[3],                 // low
                        parts[4],                 // close
                        parts[5],                 // volume
                        Long.parseLong(parts[6]), // closeTime
                        parts[7],                 // quoteAssetVolume
                        Integer.parseInt(parts[8]),// numberOfTrades
                        parts[9],                 // takerBuyBaseAssetVolume
                        parts[10],                // takerBuyQuoteAssetVolume
                        parts[11]                 // ignore
                    ));
                } catch (NumberFormatException e) {
                    System.err.println("Skipping kline CSV line due to parse error: " + line + " | Error: " + e.getMessage());
                }
            }
        } // try-with-resources will close the reader
        return klines;
    }

    private List<BinanceTrade> parseTradesFromCsv(Path csvFile) throws IOException {
        List<BinanceTrade> trades = new ArrayList<>();
         if (!Files.exists(csvFile) || Files.size(csvFile) == 0) {
            System.out.println("CSV file for trades is empty or does not exist: " + csvFile);
            return trades;
        }

        try (IndexedCsvReader reader = new IndexedCsvReader(csvFile)) {
            // String header = reader.getHeader(); // Optional
            for (String line : reader) { // Iterate using the Iterable interface
                if (line.trim().isEmpty()) continue;
                String[] parts = line.split(",");
                if (parts.length < 7) {
                    System.err.println("Skipping trade CSV line due to insufficient parts: " + line);
                    continue;
                }
                try {
                    trades.add(new BinanceTrade(
                        Long.parseLong(parts[0]),   // tradeId
                        parts[1],                   // price
                        parts[2],                   // qty
                        parts[3],                   // quoteQty
                        Long.parseLong(parts[4]),   // time
                        Boolean.parseBoolean(parts[5]), // isBuyerMaker
                        Boolean.parseBoolean(parts[6])  // isBestMatch
                    ));
                } catch (NumberFormatException e) {
                     System.err.println("Skipping trade CSV line due to parse error: " + line + " | Error: " + e.getMessage());
                }
            }
        } // try-with-resources will close the reader
        return trades;
    }
}
