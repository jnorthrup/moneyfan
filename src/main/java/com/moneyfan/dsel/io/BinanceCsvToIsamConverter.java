package com.moneyfan.dsel.io;

import com.moneyfan.dsel.D;
import com.moneyfan.dsel.core.ColumnMeta;
import com.moneyfan.dsel.core.RowVec;
import com.moneyfan.dsel.core.TypeMemento;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

public class BinanceCsvToIsamConverter {

    private static final String META_SFX = ".meta";
    private static final String DATA_SFX = ".dat";
    private static final String DEFAULT_DELIMITER = ",";

    /**
     * Converts a Binance K-line CSV file into an ISAM data file and its metadata file.
     * Assumes a fixed schema for Binance K-line data.
     *
     * @param csvFilePath The path to the input CSV file.
     * @param outputBaseName The base name for the output ISAM files (e.g., "BTCUSDT_1m").
     *                       This will create "outputBaseName.meta" and "outputBaseName.dat".
     * @throws IOException If an I/O error occurs during file operations.
     * @throws IllegalArgumentException If the CSV format does not match the expected schema
     *                                  or if the file is empty.
     */
    public static void convert(String csvFilePath, String outputBaseName) throws IOException {
        Objects.requireNonNull(csvFilePath, "CSV file path cannot be null");
        Objects.requireNonNull(outputBaseName, "Output base name cannot be null");

        // Define the expected schema for Binance K-line data
        // Based on typical Binance K-line CSV format:
        // Open time,Open,High,Low,Close,Volume,Close time,Quote asset volume,Number of trades,Taker buy base asset volume,Taker buy quote asset volume,Ignore
        List<ColumnMeta> schema = Arrays.asList(
                D.createColumnMeta("Open time", TypeMemento.Basic.LONG), // Unix timestamp in milliseconds
                D.createColumnMeta("Open", TypeMemento.Basic.DOUBLE),
                D.createColumnMeta("High", TypeMemento.Basic.DOUBLE),
                D.createColumnMeta("Low", TypeMemento.Basic.DOUBLE),
                D.createColumnMeta("Close", TypeMemento.Basic.DOUBLE),
                D.createColumnMeta("Volume", TypeMemento.Basic.DOUBLE),
                D.createColumnMeta("Close time", TypeMemento.Basic.LONG), // Unix timestamp in milliseconds
                D.createColumnMeta("Quote asset volume", TypeMemento.Basic.DOUBLE),
                D.createColumnMeta("Number of trades", TypeMemento.Basic.LONG),
                D.createColumnMeta("Taker buy base asset volume", TypeMemento.Basic.DOUBLE),
                D.createColumnMeta("Taker buy quote asset volume", TypeMemento.Basic.DOUBLE),
                D.createColumnMeta("Ignore", TypeMemento.Basic.DOUBLE) // Often 0 or ignored, can be STRING or other if needed
        );

        // Calculate recordByteLength once from the schema using IsamFileMetadata's constructor logic
        D.IsamFileMetadata initialMetadata = new D.IsamFileMetadata(schema, 0);
        int recordByteLength = initialMetadata.recordByteLength();

        long recordCount = 0;
        String dataFilePath = outputBaseName + DATA_SFX;
        String metaFilePath = outputBaseName + META_SFX;

        try (BufferedReader reader = new BufferedReader(new FileReader(csvFilePath));
             RandomAccessFile dataFile = new RandomAccessFile(dataFilePath, "rw")) {

            // Skip header line
            String headerLine = reader.readLine();
            if (headerLine == null) {
                throw new IOException("CSV file is empty: " + csvFilePath);
            }

            // Prepare ByteBuffer for writing each row
            byte[] rowBytes = new byte[recordByteLength];
            ByteBuffer bb = ByteBuffer.wrap(rowBytes).order(ByteOrder.BIG_ENDIAN); // Use BIG_ENDIAN for network order consistency

            String line;
            while ((line = reader.readLine()) != null) {
                if (line.trim().isEmpty()) continue; // Skip empty lines

                // Parse CSV line into a RowVec using the defined schema
                RowVec rowVec = D.parseCsvLine(line, schema, DEFAULT_DELIMITER);

                bb.clear(); // Reset buffer position to 0 and limit to capacity for new row
                // Write each value from the RowVec into the ByteBuffer
                for (int i = 0; i < schema.size(); i++) {
                    Object value = D.get(rowVec, i); // Get value from RowVec
                    ColumnMeta cm = schema.get(i); // Get ColumnMeta for type information
                    D.writeValueToBuffer(bb, value, cm.s()); // Write value to buffer using its TypeMemento
                }

                // Write the filled byte array to the data file
                dataFile.write(rowBytes);
                recordCount++;
            }
        }

        // Create final metadata with actual record count and the pre-calculated recordByteLength
        D.IsamFileMetadata finalMetadata = new D.IsamFileMetadata(schema, recordCount, recordByteLength);
        try (DataOutputStream metaOut = new DataOutputStream(new FileOutputStream(metaFilePath))) {
            finalMetadata.write(metaOut); // Write the metadata to the .meta file
        }

        System.out.println(String.format("Successfully converted '%s' to ISAM files '%s.meta' and '%s.dat'. Records: %d, Record Length: %d bytes.",
                csvFilePath, outputBaseName, outputBaseName, recordCount, finalMetadata.recordByteLength()));
    }
}
