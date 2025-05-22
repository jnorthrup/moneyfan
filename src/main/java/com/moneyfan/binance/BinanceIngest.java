package com.moneyfan.binance;

import com.moneyfan.core.IOMemento;
import com.moneyfan.core.Scalar;
import com.moneyfan.grid.GridCursor;
import com.moneyfan.grid.RowVec;
import com.moneyfan.io.CSVCursorReader;
import com.moneyfan.io.ISAMWriter;

import java.io.IOException;
import java.nio.file.Path;
import java.time.Instant;
import java.util.List;
import java.util.function.Predicate;

/**
 * Sample class for Binance data ingest.
 */
public class BinanceIngest {
    
    // Define Binance kline CSV schema
    public static final List<Scalar> BINANCE_KLINE_SCHEMA = List.of(
        Scalar.of(IOMemento.IO_LONG, "open_time"),             // Open time
        Scalar.of(IOMemento.IO_DOUBLE, "open"),                // Open price
        Scalar.of(IOMemento.IO_DOUBLE, "high"),                // High price
        Scalar.of(IOMemento.IO_DOUBLE, "low"),                 // Low price
        Scalar.of(IOMemento.IO_DOUBLE, "close"),               // Close price
        Scalar.of(IOMemento.IO_DOUBLE, "volume"),              // Volume
        Scalar.of(IOMemento.IO_LONG, "close_time"),            // Close time
        Scalar.of(IOMemento.IO_DOUBLE, "quote_asset_volume"),  // Quote asset volume
        Scalar.of(IOMemento.IO_LONG, "number_of_trades"),      // Number of trades
        Scalar.of(IOMemento.IO_DOUBLE, "taker_buy_base_vol"),  // Taker buy base asset volume
        Scalar.of(IOMemento.IO_DOUBLE, "taker_buy_quote_vol"), // Taker buy quote asset volume
        Scalar.stringOf("ignore", 8)                           // Ignore (can be empty string)
    );
    
    public static void processKlineData(Path csvPath, Path dataPath, Path metaPath) throws IOException {
        // 1. Read CSV into GridCursor
        System.out.println("Reading CSV from: " + csvPath);
        GridCursor cursor = CSVCursorReader.readFromCSV(csvPath, BINANCE_KLINE_SCHEMA, true);
        
        // 2. Optional: Apply transformations (e.g., convert timestamps to Instant objects)
        GridCursor transformedCursor = transformTimestamps(cursor);
        
        // 3. Write to ISAM format
        System.out.println("Writing ISAM data to: " + dataPath);
        System.out.println("Writing ISAM metadata to: " + metaPath);
        ISAMWriter.writeGridCursor(transformedCursor, dataPath, metaPath);
        
        System.out.println("Processed " + cursor.rowCount() + " rows");
    }
    
    private static GridCursor transformTimestamps(GridCursor cursor) {
        // Convert timestamp columns to Instant objects
        return cursor.filter(filterValidRows());
    }
    
    private static Predicate<RowVec> filterValidRows() {
        return row -> {
            // Example filter: Ensure volume is > 0
            Double volume = (Double) row.getValue(5);
            return volume != null && volume > 0;
        };
    }
    
    // Example agent data provider interface
    public interface AgentDataProvider {
        RowVec getNextDataPoint();
    }
    
    // Sample implementation of agent data provider
    public static class BinanceAgentDataProvider implements AgentDataProvider {
        private final GridCursor cursor;
        private int currentIndex = 0;
        
        public BinanceAgentDataProvider(GridCursor cursor) {
            this.cursor = cursor;
        }
        
        @Override
        public RowVec getNextDataPoint() {
            if (currentIndex >= cursor.rowCount()) {
                return null; // End of data
            }
            return cursor.getRow(currentIndex++);
        }
    }
    
    public static void main(String[] args) {
        if (args.length < 3) {
            System.out.println("Usage: BinanceIngest <csvPath> <dataPath> <metaPath>");
            return;
        }
        
        try {
            processKlineData(
                Path.of(args[0]), 
                Path.of(args[1]), 
                Path.of(args[2])
            );
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}