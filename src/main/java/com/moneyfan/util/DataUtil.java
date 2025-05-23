package com.moneyfan.util;

import com.moneyfan.dsel.D;
import com.moneyfan.dsel.core.ColumnMeta;
import com.moneyfan.dsel.core.TypeMemento;
import com.moneyfan.simulator.model.AssetKey;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

public enum DataUtil {
    ;
    // Schema for Kline: Timestamp(L), Open(D), High(D), Low(D), Close(D), Volume(D)
    public static final List<ColumnMeta> KLINE_SCHEMA = List.of(
        D.cm("Timestamp", TypeMemento.Basic.LONG),
        D.cm("Open", TypeMemento.Basic.DOUBLE),
        D.cm("High", TypeMemento.Basic.DOUBLE),
        D.cm("Low", TypeMemento.Basic.DOUBLE),
        D.cm("Close", TypeMemento.Basic.DOUBLE),
        D.cm("Volume", TypeMemento.Basic.DOUBLE)
    );

    public static Path ensureDirs(String baseDir, AssetKey assetKey) throws IOException {
        Path dir = Paths.get(baseDir, "klines", "1m", assetKey.baseAsset(), assetKey.quoteAsset());
        Files.createDirectories(dir);
        return dir;
    }

    public static void generateDummyCsv(Path filePath, long numRecords) throws IOException {
        Random random = ThreadLocalRandom.current();
        Instant startTime = Instant.now().minus(numRecords, ChronoUnit.MINUTES);

        try (PrintWriter writer = new PrintWriter(Files.newBufferedWriter(filePath))) {
            writer.println("Timestamp,Open,High,Low,Close,Volume"); // Header
            double lastClose = 100.0 + random.nextDouble(-10, 10);
            for (long i = 0; i < numRecords; i++) {
                long timestamp = startTime.plus(i, ChronoUnit.MINUTES).toEpochMilli();
                double open = lastClose * (1 + random.nextDouble(-0.005, 0.005)); // open around last close
                double high = Math.max(open, lastClose) * (1 + random.nextDouble(0, 0.005));
                double low = Math.min(open, lastClose) * (1 - random.nextDouble(0, 0.005));
                double close = low + random.nextDouble(high - low);
                double volume = 10 + random.nextDouble(100);
                writer.printf("%d,%.2f,%.2f,%.2f,%.2f,%.2f\n", timestamp, open, high, low, close, volume);
                lastClose = close;
            }
        }
    }

    public static final List<AssetKey> NINETEEN_PAIRS = List.of(
        AssetKey.of("BTC/USDT"), AssetKey.of("ETH/USDT"), AssetKey.of("BNB/USDT"), AssetKey.of("ADA/USDT"),
        AssetKey.of("XRP/USDT"), AssetKey.of("SOL/USDT"), AssetKey.of("DOT/USDT"), AssetKey.of("DOGE/USDT"),
        AssetKey.of("AVAX/USDT"), AssetKey.of("SHIB/USDT"), AssetKey.of("MATIC/USDT"), AssetKey.of("LTC/USDT"),
        AssetKey.of("UNI/USDT"), AssetKey.of("LINK/USDT"), AssetKey.of("TRX/USDT"), AssetKey.of("BCH/USDT"),
        AssetKey.of("ALGO/USDT"), AssetKey.of("XLM/USDT"), AssetKey.of("ATOM/USDT")
    );

    public static void prepareDataForPairs(String baseImportDir, String baseIsamDir, List<AssetKey> pairs, long recordsPerPair) throws IOException {
        System.out.println("Preparing data for " + pairs.size() + " pairs...");
        for (AssetKey pair : pairs) {
            Path csvDir = ensureDirs(baseImportDir, pair);
            Path csvFile = csvDir.resolve("final-" + pair.baseAsset() + "-" + pair.quoteAsset() + "-1m.csv");
            generateDummyCsv(csvFile, recordsPerPair);
            System.out.println("Generated dummy CSV: " + csvFile);

            Path isamDir = ensureDirs(baseIsamDir, pair); // ISAM typically in a separate "processed" dir
            String isamPathBase = isamDir.resolve("final-" + pair.baseAsset() + "-" + pair.quoteAsset() + "-1m").toString();
            D.csvToIsam(csvFile.toString(), isamPathBase, KLINE_SCHEMA, ",", true);
            System.out.println("Converted to ISAM: " + isamPathBase);
        }
        System.out.println("Data preparation complete.");
    }
}
