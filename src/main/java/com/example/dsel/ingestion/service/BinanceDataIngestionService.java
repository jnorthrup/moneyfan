package com.example.dsel.ingestion.service;

import com.example.dsel.ingestion.config.AppConfig;
import com.example.dsel.ingestion.dto.BinanceKline;
import com.example.dsel.ingestion.dto.BinanceTrade;
import com.example.dsel.ingestion.schema.DselSchemas; // To access the actual schema lists
import borg.trikeshed.cursor.Cursor;
import borg.trikeshed.lib.Series; // For Series.size() if needed on Cursor

import java.util.List;

public class BinanceDataIngestionService {

    private final AppConfig appConfig;
    private final BinanceApiWrapper binanceApiWrapper;
    private final DataTransformerService dataTransformerService;
    private final IncrementalUpdateManager incrementalUpdateManager;
    private final IsamPersistenceService isamPersistenceService;

    public BinanceDataIngestionService(AppConfig appConfig,
                                       BinanceApiWrapper binanceApiWrapper,
                                       DataTransformerService dataTransformerService,
                                       IncrementalUpdateManager incrementalUpdateManager,
                                       IsamPersistenceService isamPersistenceService) {
        this.appConfig = appConfig;
        this.binanceApiWrapper = binanceApiWrapper;
        this.dataTransformerService = dataTransformerService;
        this.incrementalUpdateManager = incrementalUpdateManager;
        this.isamPersistenceService = isamPersistenceService;
    }

    public void ingestAllData() {
        List<String> assets = appConfig.getTrackedAssets();
        List<String> intervals = appConfig.getTargetTimeUnits();
        int apiCallLimit = 1000; // Common limit for Binance API

        for (String assetPair : assets) {
            for (String interval : intervals) {
                // Ingest Klines
                ingestDataType(assetPair, interval, "klines", apiCallLimit);
                // Ingest Trades (if desired, can be a separate loop or conditional)
                // For now, focusing on klines based on most examples, trades are similar
                // ingestDataType(assetPair, interval, "trades", apiCallLimit); 
            }
        }
        System.out.println("Data ingestion cycle complete.");
    }

    private void ingestDataType(String assetPair, String interval, String dataType, int limit) {
        System.out.printf("Starting ingestion for %s - %s - %s%n", assetPair, interval, dataType);
        try {
            long lastTimestamp = incrementalUpdateManager.getLastTimestamp(assetPair, interval, dataType);
            // Start time for API call should be lastTimestamp + 1 millisecond for klines to avoid re-fetching the last candle.
            // For trades, lastTimestamp could be used directly if trade IDs are the primary mechanism for preventing duplicates.
            // Binance kline startTime is inclusive.
            long nextStartTime = (dataType.equals("klines") && lastTimestamp > 0) ? lastTimestamp + 1 : lastTimestamp; 

            // endTime can be current time, or often APIs fetch up to most recent available if endTime is far in future or null.
            // For simplicity in this placeholder, we'll imagine fetching one batch.
            // A real implementation would loop until no more data is returned by the API for the period.
            long hypotheticalEndTime = System.currentTimeMillis(); // Or a more sophisticated range management

            if (dataType.equals("klines")) {
                List<BinanceKline> klines = binanceApiWrapper.getKlines(assetPair, interval, nextStartTime, hypotheticalEndTime, limit);
                if (klines != null && !klines.isEmpty()) {
                    Cursor klineCursor = dataTransformerService.transformKlines(klines, DselSchemas.KLINE_SCHEMA);
                    if (klineCursor != null && klineCursor.size() > 0) {
                        isamPersistenceService.saveCursor(klineCursor, assetPair, interval, dataType, DselSchemas.KLINE_SCHEMA);
                        // Update last timestamp with the closeTime of the last kline in the batch
                        // Assuming klines are sorted by time ascending
                        long newLastTimestamp = klines.get(klines.size() - 1).closeTime(); 
                        incrementalUpdateManager.updateLastTimestamp(assetPair, interval, dataType, newLastTimestamp);
                        System.out.printf("Successfully ingested %d klines for %s - %s. New last timestamp: %d%n", klines.size(), assetPair, interval, newLastTimestamp);
                    } else {
                         System.out.printf("No klines transformed for %s - %s.%n", assetPair, interval);
                    }
                } else {
                    System.out.printf("No klines fetched from API for %s - %s for startTime %d.%n", assetPair, interval, nextStartTime);
                }
            } else if (dataType.equals("trades")) {
                List<BinanceTrade> trades = binanceApiWrapper.getTrades(assetPair, nextStartTime, hypotheticalEndTime, limit);
                if (trades != null && !trades.isEmpty()) {
                    Cursor tradeCursor = dataTransformerService.transformTrades(trades, DselSchemas.TRADE_SCHEMA);
                     if (tradeCursor != null && tradeCursor.size() > 0) {
                        isamPersistenceService.saveCursor(tradeCursor, assetPair, interval, dataType, DselSchemas.TRADE_SCHEMA);
                        // Update last timestamp with the time of the last trade in the batch
                        long newLastTimestamp = trades.get(trades.size() - 1).time();
                        incrementalUpdateManager.updateLastTimestamp(assetPair, interval, dataType, newLastTimestamp);
                        System.out.printf("Successfully ingested %d trades for %s - %s. New last timestamp: %d%n", trades.size(), assetPair, interval, newLastTimestamp);
                    } else {
                        System.out.printf("No trades transformed for %s - %s.%n", assetPair, interval);
                    }
                } else {
                    System.out.printf("No trades fetched from API for %s - %s for startTime %d.%n", assetPair, interval, nextStartTime);
                }
            }
        } catch (Exception e) {
            System.err.printf("Error during ingestion for %s - %s - %s: %s%n", assetPair, interval, dataType, e.getMessage());
            e.printStackTrace();
        }
    }
}
