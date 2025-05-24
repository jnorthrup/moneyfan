package com.example.dsel.ingestion.service;

import com.example.dsel.ingestion.config.AppConfig;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.time.LocalDate;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.Comparator;

public class ExternalDataFetcher {

    private final AppConfig appConfig;
    private final String mpCachePath;
    private final String mpImportPath;
    private static final DateTimeFormatter YEAR_MONTH_FORMATTER = DateTimeFormatter.ofPattern("yyyy-MM");
    private static final DateTimeFormatter DATE_FORMATTER = DateTimeFormatter.ofPattern("yyyy-MM-dd");

    public ExternalDataFetcher(AppConfig appConfig) {
        this.appConfig = appConfig;
        this.mpCachePath = resolvePath(appConfig.getMpCacheBasePath());
        this.mpImportPath = resolvePath(appConfig.getMpImportBasePath());
    }

    private String resolvePath(String pathStr) {
        if (pathStr == null) return "."; // Default to current dir if null
        if (pathStr.startsWith("~" + java.io.File.separator) || pathStr.equals("~")) {
            return System.getProperty("user.home") + pathStr.substring(1);
        } else if (pathStr.startsWith("~")) {
            return System.getProperty("user.home");
        }
        return pathStr;
    }

    private void runCommand(List<String> command, Path workingDirectory, String description) throws IOException, InterruptedException {
        System.out.println("Running command (" + description + "): " + String.join(" ", command));
        ProcessBuilder pb = new ProcessBuilder(command);
        if (workingDirectory != null) {
            pb.directory(workingDirectory.toFile());
        }
        
        // Capture output for better error reporting if needed
        Process process = pb.start();
        
        // Read stdout
        StringBuilder stdOut = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
            String line;
            while ((line = reader.readLine()) != null) {
                stdOut.append(line).append(System.lineSeparator());
            }
        }
        // Read stderr
        StringBuilder stdErr = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getErrorStream()))) {
            String line;
            while ((line = reader.readLine()) != null) {
                stdErr.append(line).append(System.lineSeparator());
            }
        }

        int exitCode = process.waitFor();
        if (stdOut.length() > 0) {
            System.out.println("Stdout ("+description+"): \n" + stdOut);
        }
        if (stdErr.length() > 0) {
            System.err.println("Stderr ("+description+"): \n" + stdErr);
        }

        if (exitCode != 0) {
            throw new IOException("Command (" + description + ") failed with exit code " + exitCode + ": " + String.join(" ", command) + "\nError output:\n" + stdErr);
        }
    }

    public Path fetchAndProcessSpecificMonthlyKlines(String assetPair, String timeUnit, int year, int month) throws IOException, InterruptedException {
        String symbol = assetPair.replace("/", "");
        Path klineCacheDir = Paths.get(mpCachePath, "klines", timeUnit, symbol, String.format("%d-%02d", year, month)); // Subdir for month
        Files.createDirectories(klineCacheDir);

        List<String> urls = generateKlineUrls(symbol, timeUnit, year, month, year, month); // Fetch for single month
        if (urls.isEmpty()) {
            System.out.println("No monthly kline URLs generated for " + symbol + " " + timeUnit + " " + year + "-" + month);
            return null; // Or throw exception
        }

        List<String> ariaCommand = new ArrayList<>(List.of("aria2c", "-Z", "-c", "-x", "15", "-j", "15", "-s", "15", "-d", klineCacheDir.toString()));
        ariaCommand.addAll(urls);
        runCommand(ariaCommand, null, "Download monthly klines for " + symbol + " " + timeUnit + " " + year + "-" + month);

        Path tempUnzipDir = Files.createTempDirectory("unzip_monthly_klines_" + symbol + "_");
        Path processedMonthlyCsv;
        try {
            List<Path> zipFiles = Files.list(klineCacheDir)
                                     .filter(p -> p.toString().endsWith(".zip"))
                                     .collect(Collectors.toList());
            if (zipFiles.isEmpty()) {
                System.out.println("No zip files found for monthly klines " + symbol + " " + timeUnit + " " + year + "-" + month);
                return null;
            }
            for(Path zipFile : zipFiles) {
                runCommand(List.of("unzip", "-aa", "-n", zipFile.toString(), "-d", tempUnzipDir.toString()), null, "Unzip " + zipFile.getFileName());
            }
            
            // Process this single month's CSV data
            processedMonthlyCsv = klineCacheDir.resolve(String.format("%s-%s-%d-%02d-processed.csv", symbol, timeUnit, year, month));
            String header = "Open_time,Open,High,Low,Close,Volume,Close_time,Quote_asset_volume,Number_of_trades,Taker_buy_base_asset_volume,Taker_buy_quote_asset_volume,Ignore";
            // Note: The find command needs to be relative to tempUnzipDir. Using "."
            String combinedCsvProcessingCommand = String.format(
                "echo '%s' > %s && find . -name '*.csv' -print0 | xargs -0 sort -fu | grep --extended-regexp -e '(.*,){11}' | sed --posix --regexp-extended 's/(\\.[0-9]+])0+,/\\1,/g' >> %s",
                header, processedMonthlyCsv.toString(), processedMonthlyCsv.toString()
            );
            runCommand(List.of("bash", "-c", combinedCsvProcessingCommand), tempUnzipDir, "Process monthly klines for " + symbol + " " + year + "-" + month);
            return processedMonthlyCsv;
        } finally {
            Files.walk(tempUnzipDir)
                 .sorted(Comparator.reverseOrder())
                 .map(Path::toFile)
                 .forEach(java.io.File::delete);
        }
    }

    public Path fetchAndProcessSpecificDailyKlines(String assetPair, String timeUnit, LocalDate date) throws IOException, InterruptedException {
        String symbol = assetPair.replace("/", "");
        Path klineCacheDir = Paths.get(mpCachePath, "klines", timeUnit, symbol, date.format(DATE_FORMATTER)); // Subdir for date
        Files.createDirectories(klineCacheDir);

        List<String> urls = generateDailyKlineUrls(symbol, timeUnit, date, date); // Fetch for single day
        if (urls.isEmpty()) {
            System.out.println("No daily kline URLs generated for " + symbol + " " + timeUnit + " " + date);
            return null;
        }

        List<String> ariaCommand = new ArrayList<>(List.of("aria2c", "-Z", "-c", "-x", "15", "-j", "15", "-s", "15", "-d", klineCacheDir.toString()));
        ariaCommand.addAll(urls);
        runCommand(ariaCommand, null, "Download daily klines for " + symbol + " " + timeUnit + " " + date);

        Path tempUnzipDir = Files.createTempDirectory("unzip_daily_klines_" + symbol + "_");
        Path processedDailyCsv;
        try {
            List<Path> zipFiles = Files.list(klineCacheDir)
                                     .filter(p -> p.toString().endsWith(".zip"))
                                     .collect(Collectors.toList());
            if (zipFiles.isEmpty()) {
                System.out.println("No zip files found for daily klines " + symbol + " " + timeUnit + " " + date);
                return null;
            }
            for(Path zipFile : zipFiles) {
                runCommand(List.of("unzip", "-aa", "-n", zipFile.toString(), "-d", tempUnzipDir.toString()), null, "Unzip " + zipFile.getFileName());
            }
            
            processedDailyCsv = klineCacheDir.resolve(String.format("%s-%s-%s-processed.csv", symbol, timeUnit, date.format(DATE_FORMATTER)));
            String header = "Open_time,Open,High,Low,Close,Volume,Close_time,Quote_asset_volume,Number_of_trades,Taker_buy_base_asset_volume,Taker_buy_quote_asset_volume,Ignore";
            String combinedCsvProcessingCommand = String.format(
                "echo '%s' > %s && find . -name '*.csv' -print0 | xargs -0 sort -fu | grep --extended-regexp -e '(.*,){11}' | sed --posix --regexp-extended 's/(\\.[0-9]+])0+,/\\1,/g' >> %s",
                header, processedDailyCsv.toString(), processedDailyCsv.toString()
            );
            runCommand(List.of("bash", "-c", combinedCsvProcessingCommand), tempUnzipDir, "Process daily klines for " + symbol + " " + date);
            return processedDailyCsv;
        } finally {
            Files.walk(tempUnzipDir)
                 .sorted(Comparator.reverseOrder())
                 .map(Path::toFile)
                 .forEach(java.io.File::delete);
        }
    }

    // --- KLINE PROCESSING ---

    public Path fetchAndProcessKlines(String assetPair, String timeUnit) throws IOException, InterruptedException {
        String symbol = assetPair.replace("/", "");
        Path klineCacheDir = Paths.get(mpCachePath, "klines", timeUnit, symbol, symbol); // Matches script structure like $MP_CACHE/klines/$TUNIT/$TC/$CC
        Path targetDir = Paths.get(mpImportPath, "klines", timeUnit, symbol, symbol);    // Matches script structure like $MP_IMPORT/klines/$TUNIT/$TC/$CC
        Files.createDirectories(klineCacheDir);
        Files.createDirectories(targetDir);

        // For initial bulk load, fetch a wide range of monthly data.
        // E.g., from 2017-01 to current year's previous month.
        LocalDate today = LocalDate.now(ZoneOffset.UTC);
        LocalDate lastMonth = today.minusMonths(1);
        List<String> urls = generateKlineUrls(symbol, timeUnit, 2017, 1, lastMonth.getYear(), lastMonth.getMonthValue());
        
        // If fetching for very recent past, might need daily for current month too.
        // urls.addAll(generateDailyKlineUrls(symbol, timeUnit, today.withDayOfMonth(1), today)); // Example: current month daily

        List<String> ariaCommand = new ArrayList<>(List.of("aria2c", "-Z", "-c", "-x", "15", "-j", "15", "-s", "15", "-d", klineCacheDir.toString()));
        ariaCommand.addAll(urls);
        if (!urls.isEmpty()) {
            runCommand(ariaCommand, null, "Download klines for " + symbol + " " + timeUnit);
        } else {
            System.out.println("No URLs generated for initial bulk kline download for " + symbol + " " + timeUnit);
        }
        
        Path tempUnzipDir = Files.createTempDirectory("unzip_klines_" + symbol + "_");
        try {
            // Unzip all files from klineCacheDir into tempUnzipDir
            // find klineCacheDir -name "*.zip" -exec unzip -o -d tempUnzipDir {} \;
            // More robust: list files and unzip one by one or use a Java library if complex patterns
            List<Path> zipFiles = Files.list(klineCacheDir)
                                     .filter(p -> p.toString().endsWith(".zip"))
                                     .collect(Collectors.toList());
            for(Path zipFile : zipFiles) {
                runCommand(List.of("unzip", "-aa", "-n", zipFile.toString(), "-d", tempUnzipDir.toString()), null, "Unzip " + zipFile.getFileName());
            }
            
            Path finalCsvFile = targetDir.resolve("final-" + symbol + "-" + timeUnit + ".csv");
            
            // Combine, sort, and process CSVs
            // This replicates: echo 'header'; sort -fu *.csv | grep | sed > final.csv
            // Using a shell command for this complex part as it's much simpler than Java equivalent
            String header = "Open_time,Open,High,Low,Close,Volume,Close_time,Quote_asset_volume,Number_of_trades,Taker_buy_base_asset_volume,Taker_buy_quote_asset_volume,Ignore";
            String combinedCsvProcessingCommand = String.format(
                "echo '%s' > %s && find . -name '*.csv' -print0 | xargs -0 sort -fu | grep --extended-regexp -e '(.*,){11}' | sed --posix --regexp-extended 's/(\\.[0-9]+])0+,/\\1,/g' >> %s",
                header, finalCsvFile.toString(), finalCsvFile.toString()
            );
            // Note: The above command writes header first, then appends sorted data.
            // The original script wrote header and then piped sort output. This is equivalent for `>` then `>>`.
            runCommand(List.of("bash", "-c", combinedCsvProcessingCommand), tempUnzipDir, "Combine and sort klines for " + symbol);
            
            return finalCsvFile;
        } finally {
            // Clean up temporary directory
            Files.walk(tempUnzipDir)
                 .sorted(Comparator.reverseOrder())
                 .map(Path::toFile)
                 .forEach(java.io.File::delete);
        }
    }

    // Generates URLs for monthly klines within a specified year/month range
    public List<String> generateKlineUrls(String symbol, String timeUnit, int startYear, int startMonth, int endYear, int endMonth) {
        List<String> urls = new ArrayList<>();
        String baseUrl = "https://data.binance.vision/data/spot/";

        for (int year = startYear; year <= endYear; year++) {
            int currentStartMonth = (year == startYear) ? startMonth : 1;
            int currentEndMonth = (year == endYear) ? endMonth : 12;
            for (int month = currentStartMonth; month <= currentEndMonth; month++) {
                String monthStr = String.format("%02d", month);
                String datePart = year + "-" + monthStr;
                String fileName = String.format("%s-%s-%s", symbol, timeUnit, datePart);
                urls.add(baseUrl + "monthly/klines/" + symbol + "/" + timeUnit + "/" + fileName + ".zip");
                urls.add(baseUrl + "monthly/klines/" + symbol + "/" + timeUnit + "/" + fileName + ".zip.CHECKSUM");
            }
        }
        return urls;
    }
    
    // Generates URLs for daily klines for a specific date or a range of dates
    public List<String> generateDailyKlineUrls(String symbol, String timeUnit, LocalDate startDate, LocalDate endDate) {
        List<String> urls = new ArrayList<>();
        String baseUrl = "https://data.binance.vision/data/spot/";
        LocalDate currentDate = startDate;
        while (!currentDate.isAfter(endDate)) {
            String datePart = currentDate.format(DATE_FORMATTER);
            String fileName = String.format("%s-%s-%s", symbol, timeUnit, datePart);
            urls.add(baseUrl + "daily/klines/" + symbol + "/" + timeUnit + "/" + fileName + ".zip");
            urls.add(baseUrl + "daily/klines/" + symbol + "/" + timeUnit + "/" + fileName + ".zip.CHECKSUM");
            currentDate = currentDate.plusDays(1);
        }
        return urls;
    }


    // --- TRADE PROCESSING (Similar structure, simplified for brevity) ---
    
    public Path fetchAndProcessTrades(String assetPair, String timeUnit) throws IOException, InterruptedException {
        String symbol = assetPair.replace("/", "");
        Path tradeCacheDir = Paths.get(mpCachePath, "trades", timeUnit, symbol, symbol);
        Path targetDir = Paths.get(mpImportPath, "trades", timeUnit, symbol, symbol);
        Files.createDirectories(tradeCacheDir);
        Files.createDirectories(targetDir);

        // For initial bulk load, fetch a wide range of monthly data.
        LocalDate today = LocalDate.now(ZoneOffset.UTC);
        LocalDate lastMonth = today.minusMonths(1);
        List<String> urls = generateTradeUrls(symbol, timeUnit, 2017, 1, lastMonth.getYear(), lastMonth.getMonthValue());
        // urls.addAll(generateDailyTradeUrls(symbol, timeUnit, today.withDayOfMonth(1), today)); // Example: current month daily


        List<String> ariaCommand = new ArrayList<>(List.of("aria2c", "-Z", "-c", "-x", "15", "-j", "15", "-s", "15", "-d", tradeCacheDir.toString()));
        ariaCommand.addAll(urls);
        if (!urls.isEmpty()) {
            runCommand(ariaCommand, null, "Download trades for " + symbol + " " + timeUnit);
        } else {
            System.out.println("No URLs generated for initial bulk trade download for " + symbol + " " + timeUnit);
        }
        
        Path tempUnzipDir = Files.createTempDirectory("unzip_trades_" + symbol + "_");
        try {
            List<Path> zipFiles = Files.list(tradeCacheDir)
                                     .filter(p -> p.toString().endsWith(".zip"))
                                     .collect(Collectors.toList());
            for(Path zipFile : zipFiles) {
                runCommand(List.of("unzip", "-aa", "-n", zipFile.toString(), "-d", tempUnzipDir.toString()), null, "Unzip " + zipFile.getFileName());
            }

            Path finalCsvFile = targetDir.resolve(timeUnit + ".csv"); // Script uses $TUNIT.csv
            String header = "trade_Id,price,qty,quoteQty,time,isBuyerMaker,isBestMatch"; // Matches fetchtrades.sh
            // The script uses `cat *.csv`, which doesn't sort.
            String combinedCsvProcessingCommand = String.format(
                "echo '%s' > %s && find . -name '*.csv' -print0 | xargs -0 cat >> %s",
                header, finalCsvFile.toString(), finalCsvFile.toString()
            );
             runCommand(List.of("bash", "-c", combinedCsvProcessingCommand), tempUnzipDir, "Combine trades for " + symbol);
            return finalCsvFile;
        } finally {
            Files.walk(tempUnzipDir)
                 .sorted(Comparator.reverseOrder())
                 .map(Path::toFile)
                 .forEach(java.io.File::delete);
        }
    }

    public Path fetchAndProcessSpecificMonthlyTrades(String assetPair, String timeUnit, int year, int month) throws IOException, InterruptedException {
        String symbol = assetPair.replace("/", "");
        Path tradeCacheDir = Paths.get(mpCachePath, "trades", timeUnit, symbol, String.format("%d-%02d", year, month));
        Files.createDirectories(tradeCacheDir);

        List<String> urls = generateTradeUrls(symbol, timeUnit, year, month, year, month);
        if (urls.isEmpty()) {
            System.out.println("No monthly trade URLs generated for " + symbol + " " + timeUnit + " " + year + "-" + month);
            return null;
        }

        List<String> ariaCommand = new ArrayList<>(List.of("aria2c", "-Z", "-c", "-x", "15", "-j", "15", "-s", "15", "-d", tradeCacheDir.toString()));
        ariaCommand.addAll(urls);
        runCommand(ariaCommand, null, "Download monthly trades for " + symbol + " " + timeUnit + " " + year + "-" + month);
        
        Path tempUnzipDir = Files.createTempDirectory("unzip_monthly_trades_" + symbol + "_");
        Path processedMonthlyCsv;
        try {
            List<Path> zipFiles = Files.list(tradeCacheDir)
                                     .filter(p -> p.toString().endsWith(".zip"))
                                     .collect(Collectors.toList());
            if (zipFiles.isEmpty()) {
                System.out.println("No zip files found for monthly trades " + symbol + " " + timeUnit + " " + year + "-" + month);
                return null;
            }
            for(Path zipFile : zipFiles) {
                runCommand(List.of("unzip", "-aa", "-n", zipFile.toString(), "-d", tempUnzipDir.toString()), null, "Unzip " + zipFile.getFileName());
            }

            processedMonthlyCsv = tradeCacheDir.resolve(String.format("%s-trades-%d-%02d-processed.csv", symbol, year, month));
            String header = "trade_Id,price,qty,quoteQty,time,isBuyerMaker,isBestMatch";
            String combinedCsvProcessingCommand = String.format(
                "echo '%s' > %s && find . -name '*.csv' -print0 | xargs -0 cat >> %s", // Trades are not sorted by timestamp in original script
                header, processedMonthlyCsv.toString(), processedMonthlyCsv.toString()
            );
            runCommand(List.of("bash", "-c", combinedCsvProcessingCommand), tempUnzipDir, "Process monthly trades for " + symbol + " " + year + "-" + month);
            return processedMonthlyCsv;
        } finally {
            Files.walk(tempUnzipDir)
                 .sorted(Comparator.reverseOrder())
                 .map(Path::toFile)
                 .forEach(java.io.File::delete);
        }
    }

    public Path fetchAndProcessSpecificDailyTrades(String assetPair, String timeUnit, LocalDate date) throws IOException, InterruptedException {
        String symbol = assetPair.replace("/", "");
        Path tradeCacheDir = Paths.get(mpCachePath, "trades", timeUnit, symbol, date.format(DATE_FORMATTER));
        Files.createDirectories(tradeCacheDir);

        List<String> urls = generateDailyTradeUrls(symbol, timeUnit, date, date);
        if (urls.isEmpty()) {
            System.out.println("No daily trade URLs generated for " + symbol + " " + timeUnit + " " + date);
            return null;
        }
        List<String> ariaCommand = new ArrayList<>(List.of("aria2c", "-Z", "-c", "-x", "15", "-j", "15", "-s", "15", "-d", tradeCacheDir.toString()));
        ariaCommand.addAll(urls);
        runCommand(ariaCommand, null, "Download daily trades for " + symbol + " " + timeUnit + " " + date);

        Path tempUnzipDir = Files.createTempDirectory("unzip_daily_trades_" + symbol + "_");
        Path processedDailyCsv;
        try {
            List<Path> zipFiles = Files.list(tradeCacheDir)
                                     .filter(p -> p.toString().endsWith(".zip"))
                                     .collect(Collectors.toList());
             if (zipFiles.isEmpty()) {
                System.out.println("No zip files found for daily trades " + symbol + " " + timeUnit + " " + date);
                return null;
            }
            for(Path zipFile : zipFiles) {
                runCommand(List.of("unzip", "-aa", "-n", zipFile.toString(), "-d", tempUnzipDir.toString()), null, "Unzip " + zipFile.getFileName());
            }
            
            processedDailyCsv = tradeCacheDir.resolve(String.format("%s-trades-%s-processed.csv", symbol, date.format(DATE_FORMATTER)));
            String header = "trade_Id,price,qty,quoteQty,time,isBuyerMaker,isBestMatch";
            String combinedCsvProcessingCommand = String.format(
                "echo '%s' > %s && find . -name '*.csv' -print0 | xargs -0 cat >> %s",
                header, processedDailyCsv.toString(), processedDailyCsv.toString()
            );
            runCommand(List.of("bash", "-c", combinedCsvProcessingCommand), tempUnzipDir, "Process daily trades for " + symbol + " " + date);
            return processedDailyCsv;
        } finally {
            Files.walk(tempUnzipDir)
                 .sorted(Comparator.reverseOrder())
                 .map(Path::toFile)
                 .forEach(java.io.File::delete);
        }
    }


    // Generates URLs for monthly trades within a specified year/month range
    public List<String> generateTradeUrls(String symbol, String timeUnit, int startYear, int startMonth, int endYear, int endMonth) {
        List<String> urls = new ArrayList<>();
        String baseUrl = "https://data.binance.vision/data/spot/";

        for (int year = startYear; year <= endYear; year++) {
            int currentStartMonth = (year == startYear) ? startMonth : 1;
            int currentEndMonth = (year == endYear) ? endMonth : 12;
            for (int month = currentStartMonth; month <= currentEndMonth; month++) {
                String monthStr = String.format("%02d", month);
                String datePart = year + "-" + monthStr;
                // Trades filenames do not include timeUnit in the same way klines do with "BTCUSDT-1m-2020-01"
                // They are usually just "BTCUSDT-trades-2020-01"
                String fileName = String.format("%s-trades-%s", symbol, datePart); 
                urls.add(baseUrl + "monthly/trades/" + symbol + "/" + fileName + ".zip");
                urls.add(baseUrl + "monthly/trades/" + symbol + "/" + fileName + ".zip.CHECKSUM");
            }
        }
        return urls;
    }

    // Generates URLs for daily trades for a specific date or a range of dates
    public List<String> generateDailyTradeUrls(String symbol, String timeUnit, LocalDate startDate, LocalDate endDate) {
        List<String> urls = new ArrayList<>();
        String baseUrl = "https://data.binance.vision/data/spot/";
        LocalDate currentDate = startDate;
        while (!currentDate.isAfter(endDate)) {
            String datePart = currentDate.format(DATE_FORMATTER);
            String fileName = String.format("%s-trades-%s", symbol, datePart);
            urls.add(baseUrl + "daily/trades/" + symbol + "/" + fileName + ".zip");
            urls.add(baseUrl + "daily/trades/" + symbol + "/" + fileName + ".zip.CHECKSUM");
            currentDate = currentDate.plusDays(1);
        }
        return urls;
    }

    public Path mergeAndSortCsvFiles(Path mainCsvFile, Path newDataCsvFile, String header, Path tempDir) throws IOException, InterruptedException {
        if (newDataCsvFile == null || !Files.exists(newDataCsvFile) || Files.size(newDataCsvFile) == 0) {
            System.out.println("New data CSV is null, empty or does not exist: " + newDataCsvFile + ". No merge needed.");
            if (!Files.exists(mainCsvFile) && newDataCsvFile != null && Files.exists(newDataCsvFile)) { // If main doesn't exist but new one does (even if empty)
                 // This case is tricky, if new data is empty, we might not want to create an empty main file.
                 // For now, if new data is empty, and main doesn't exist, we do nothing with main.
            }
            return mainCsvFile; // Return mainCsvFile, which might not exist if newDataCsvFile was also null/empty
        }

        if (!Files.exists(mainCsvFile) || Files.size(mainCsvFile) == 0) {
            System.out.println("Main CSV file does not exist or is empty. Copying new data to main file: " + mainCsvFile);
            // Ensure header is written if mainCsvFile is new or empty
            // The processing steps for daily/monthly already ensure header in newDataCsvFile
            Files.copy(newDataCsvFile, mainCsvFile, StandardCopyOption.REPLACE_EXISTING);
            return mainCsvFile;
        }

        // Both files exist and have content. Time to merge.
        // Using a temporary file for the merged output before replacing the main file.
        Path tempMergedFile = tempDir.resolve(mainCsvFile.getFileName() + ".merged_temp.csv");

        // Command: cat main.csv <(tail -n +2 new_data.csv) | sort -fu -t, -k1,1 > temp_merged.csv && mv temp_merged.csv main.csv
        // Timestamps are usually in the first column for klines (Open_time) and fifth for trades (time).
        // Assuming klines for now, first column sort. For trades, this sort key might need to change or sort might be omitted.
        // The prompt specified sort -fu -t, -k1,1. This is for klines.
        // For trades, the original processing uses `cat` without sort.
        // Let's make sort key configurable or conditional. For now, using k1,1.
        // A more robust way is to check data type (klines/trades) to decide on sort.
        // For this subtask, I'll stick to the kline-style sort as per the `mergeAndSortCsvFiles` example.

        String sortColumnKey = "1,1"; // Default for klines (Open_time)
        // If we knew this was for trades, it might be "5,5" for trade time, or no sort.
        // The current `fetchAndProcessTrades` and `fetchAndProcessSpecific...Trades` methods do *not* sort the combined CSV.
        // The `mergeAndSortCsvFiles` method implies sorting is desired.
        // For now, this method will sort. If trades shouldn't be sorted, logic needs to adapt.

        String command = String.format(
            "cat %s <(tail -n +2 %s) | sort -fu -t, -k%s > %s && mv %s %s",
            mainCsvFile.toString(),
            newDataCsvFile.toString(),
            sortColumnKey,
            tempMergedFile.toString(),
            tempMergedFile.toString(),
            mainCsvFile.toString()
        );

        System.out.println("Merging " + newDataCsvFile + " into " + mainCsvFile);
        runCommand(List.of("bash", "-c", command), null, "Merge and sort CSV files");
        
        return mainCsvFile;
    }

}
