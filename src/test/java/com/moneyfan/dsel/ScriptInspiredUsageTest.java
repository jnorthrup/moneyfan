package com.moneyfan.dsel;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.stream.Collectors;

import static com.moneyfan.dsel.D.*;
import static org.junit.jupiter.api.Assertions.*;

public class ScriptInspiredUsageTest {

    @TempDir
    Path tempDir;

    private Path getResourcePath(String resourceName) {
        try {
            return Paths.get(Objects.requireNonNull(getClass().getResource(resourceName)).toURI());
        } catch (Exception e) {
            throw new RuntimeException("Cannot find resource: " + resourceName, e);
        }
    }

    /**
     * Simulates processing klines data similar to what fetchklines.sh might produce.
     * The script itself does fetching, unzipping, sorting, and combining.
     * This test focuses on what DSEL can do with the *resulting* CSV.
     */
    @Test
    void processKlinesData() throws IOException {
        // Path to a sample klines CSV (could be created by fetchklines.sh)
        Path klinesCsvPath = getResourcePath("klines_sample/import/klines/1m/BTC/USDT/final-BTC-USDT-1m.csv");

        // 1. Read the CSV using DSEL
        //    The script has a header: Open_time,Open,High,Low,Close,Volume,...
        //    DSEL will deduce types.
        Cursor klinesCursor = readCsv(klinesCsvPath.toString(), true, ',', '"', -1); // Sample all lines for type deduction

        assertNotNull(klinesCursor);
        assertTrue(sz(klinesCursor) > 0, "Klines cursor should have data");

        RowVec firstRow = get(klinesCursor, 0);
        assertEquals("Open_time", colName(firstRow, 0));
        assertEquals(TypeMemento.Basic.LONG.getTypeName(), colType(firstRow, 0).getTypeName()); // Deduced as Long
        assertEquals(1609459200000L, get(firstRow, 0));

        assertEquals("Open", colName(firstRow, 1));
        assertEquals(TypeMemento.Basic.DOUBLE.getTypeName(), colType(firstRow, 1).getTypeName()); // Deduced as Double
        assertEquals(29000.00, (Double) get(firstRow, 1), 0.0001);

        assertEquals("Volume", colName(firstRow, 5));
        assertEquals(TypeMemento.Basic.DOUBLE.getTypeName(), colType(firstRow, 5).getTypeName()); // Deduced as Double
        assertEquals(10.5, (Double) get(firstRow, 5), 0.0001);


        // 2. Perform some DSEL operations (examples)
        //    Filter klines with Volume > 10
        // Using the new get(RowVec, String) method
        Cursor filteredByVolume = fltRow(klinesCursor, row -> (Double) get(row, "Volume") > 12.0);
        assertEquals(2, sz(filteredByVolume)); // Rows with 12.3 and 15.0 volume
        assertEquals(12.3, (Double) get(get(filteredByVolume,0), "Volume"), 0.001);

        // Map to get only Close price and Volume
        final int closeIdx = 4; // Index of "Close"
        Cursor closeAndVolume = mapRow(klinesCursor, row -> {
            Object closePrice = get(row, closeIdx);
            Object volume = get(row, "Volume"); // Use name for Volume
            ColumnMeta cmClose = cm("Close", colType(row, closeIdx));
            ColumnMeta cmVolume = cm("Volume", colType(row, "Volume")); // Use name for Volume's type
            return rv(2, idx -> idx == 0 ? jn(closePrice, () -> cmClose) : jn(volume, () -> cmVolume) );
        });
        assertEquals(sz(klinesCursor), sz(closeAndVolume));
        assertEquals(2, sz(get(closeAndVolume, 0)));
        assertEquals("Close", colName(get(closeAndVolume,0),0));
        assertEquals(28999.50, (Double)get(get(closeAndVolume,0),0), 0.001);


        // 3. Convert the (original) klines data to ISAM
        String isamBasePath = tempDir.resolve("klinesIsam").toString();
        List<ColumnMeta> klinesSchemaForIsam = ls(firstRow).stream() // Use schema from the first row of the read CSV
            .map(cellJoin -> cm(f(s(cellJoin).get()), s(s(cellJoin).get())))
            .collect(Collectors.toList());

        // We need to ensure fsString has a defined length from deduced type, or override.
        // readCsv already makes string types as FixedSizeTypeMemento
        
        System.out.println("Schema for ISAM: " + klinesSchemaForIsam);
        klinesSchemaForIsam.forEach(cm -> {
            if (cm.s().getTypeName().equals(TypeMemento.Basic.STRING.getTypeName()) && cm.s().getFixedSize() <=0) {
                 throw new IllegalStateException("String column " + cm.f() + " needs fixed size for ISAM from deduction.");
            }
        });


        csvToIsam(klinesCsvPath.toString(), isamBasePath, klinesSchemaForIsam, true, ',', '"');

        // Verify ISAM can be read
        try (IsamCursor isamKlines = new IsamCursor(isamBasePath)) {
            assertEquals(sz(klinesCursor), sz(isamKlines));
            assertEquals(get(klinesCursor, 0, 0), get(get(isamKlines, 0), 0)); // Compare Open_time
            // Note: Floating point comparisons need care due to precision.
            // String representation from CSV vs binary double from ISAM.
            // D.convertCsvStringToTypedValue and D.writeValueToBuffer/readValueFromBuffer handle this.
            assertEquals((Double)get(klinesCursor, 0, 1), (Double)get(get(isamKlines, 0), 1), 0.00001); // Compare Open price
        }
        System.out.println("Klines processing example finished. ISAM created at: " + isamBasePath);
    }

    /**
     * dayklines.sh appends to a CSV. DSEL doesn't directly append to CSVs.
     * This example shows how one might take two Cursors (representing old and new data)
     * and create a combined Cursor.
     * This is more advanced Series manipulation.
     */
    @Test
    void combineCursorsExample() throws IOException {
        // Simulate "old" data
        Path oldCsvPath = tempDir.resolve("old_data.csv");
        Files.write(oldCsvPath, Arrays.asList(
            "ID,Value",
            "1,Alpha",
            "2,Beta"
        ));
        Cursor oldCursor = readCsv(oldCsvPath.toString(), true, ',', '"', 10);

        // Simulate "new" data (like from the curl command in dayklines.sh)
        Path newCsvPath = tempDir.resolve("new_data.csv");
        Files.write(newCsvPath, Arrays.asList( // New data might not have a header if appended
            "3,Gamma",
            "4,Delta"
        ));
        // Read new data, assuming same schema structure (no header means DSEL makes col_0, col_1)
        // For a true combine, we'd need to ensure schemas align or project new data to old schema.
        // Let's assume new_data.csv also had a header for simplicity here, or we provide schema.
        Files.delete(newCsvPath);
         Files.write(newCsvPath, Arrays.asList(
            "ID,Value", // Add header to make it compatible for simple read
            "3,Gamma",
            "4,Delta"
        ));
        Cursor newCursor = readCsv(newCsvPath.toString(), true, ',', '"', 10);

        // Combine: Create a new Series that delegates to old then new
        int oldSize = sz(oldCursor);
        int newSize = sz(newCursor);
        Series<RowVec> combinedSeries = sr(oldSize + newSize, i -> {
            if (i < oldSize) {
                return get(oldCursor, i);
            } else {
                return get(newCursor, i - oldSize);
            }
        });
        Cursor combinedCursor = (Cursor) combinedSeries; // Cast if Series<RowVec> is recognized as Cursor

        assertEquals(4, sz(combinedCursor));
        assertEquals(1L, get(get(combinedCursor, 0), "ID"));
        assertEquals("Alpha", get(get(combinedCursor, 0), "Value"));
        assertEquals(3L, get(get(combinedCursor, 2), "ID"));
        assertEquals("Gamma", get(get(combinedCursor, 2), "Value"));

        System.out.println("Combined cursor data:");
        each(combinedCursor, row -> {
            System.out.println(String.format("ID: %s, Value: %s", get(row, 0), get(row, 1)));
        });
    }

    /**
     * allcachedpairs.sh: find import/ -mindepth 4 -maxdepth 4 -type d |cut  -d / -f4-5
     * This script finds directory paths. Java can do this with Files.walk.
     * DSEL would then be used if these directories contained CSV or ISAM files to process.
     */
    @Test
    void simulateDirectoryTraversalAndProcess() throws IOException {
        // Setup a dummy directory structure similar to mpdata/import/klines/1m/BTC/USDT/
        Path baseImportDir = tempDir.resolve("import_sim");
        Path btcUsdtDir = baseImportDir.resolve("klines/1m/BTC/USDT");
        Files.createDirectories(btcUsdtDir);
        Path ethBtcDir = baseImportDir.resolve("klines/1m/ETH/BTC");
        Files.createDirectories(ethBtcDir);

        // Create dummy CSV files
        Files.write(btcUsdtDir.resolve("data.csv"), Arrays.asList("Pair,Price", "BTC/USDT,30000"));
        Files.write(ethBtcDir.resolve("data.csv"), Arrays.asList("Pair,Price", "ETH/BTC,0.07"));

        // Simulate `find ... | cut ...` using Java
        List<String> foundPairPaths = Files.walk(baseImportDir.resolve("klines/1m"), 2) // depth 2 from "1m"
            .filter(Files::isDirectory)
            .filter(p -> p.getParent().getFileName().toString().equals("1m") && Files.exists(p.resolve("data.csv"))) // Ensure it's TC/CC level
            .map(p -> p.getParent().getFileName().toString() + "/" + p.getFileName().toString()) // CC/TC relative to klines/1m
            .sorted()
            .collect(Collectors.toList());
        
        // The script uses -f4-5 from import/, so BTC/USDT
         foundPairPaths = Files.walk(baseImportDir, 4) 
            .filter(Files::isDirectory)
            .filter(p -> Files.exists(p.resolve("data.csv"))) // Check if it's a leaf data dir
            .map(p -> baseImportDir.relativize(p).toString()) // e.g. klines/1m/BTC/USDT
            .filter(s -> s.startsWith("klines/1m/") && s.split("/").length == 4)
            .map(s -> s.substring("klines/1m/".length())) // BTC/USDT
            .sorted()
            .collect(Collectors.toList());


        assertEquals(Arrays.asList("BTC/USDT", "ETH/BTC"), foundPairPaths);

        // Now, for each found "pair directory", process its data.csv
        for (String pairPath : foundPairPaths) {
            Path csvFile = baseImportDir.resolve("klines/1m").resolve(pairPath).resolve("data.csv");
            System.out.println("Processing: " + csvFile);
            Cursor dataCursor = readCsv(csvFile.toString(), true, ',', '"', 10);
            assertNotNull(dataCursor);
            assertEquals(1, sz(dataCursor));
            String pairName = (String) get(get(dataCursor, 0), 0);
            assertTrue(pairName.equals("BTC/USDT") || pairName.equals("ETH/BTC"));
            System.out.println("Pair: " + pairName + ", Price: " + get(get(dataCursor,0),1));
        }
    }
}
