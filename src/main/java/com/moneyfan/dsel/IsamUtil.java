package com.moneyfan.dsel;

import com.moneyfan.dsel.dsel.Join;
import com.moneyfan.dsel.dsel.functional.IOConsumer;
import com.moneyfan.dsel.dsel.functional.IOFunction;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class IsamUtil {

    // --- Conceptual ISAM Writing ---

    /**
     * Conceptually saves a TupFrame to a binary format (.bin) and a metadata file (.bin.meta).
     * This is a simplified example and not a robust ISAM implementation.
     *
     * @param filePathBase The base name for the files (e.g., "btcusdt_klines").
     * @param frame The TupFrame to save.
     * @param metaWriter Writes metadata (e.g., column names, types).
     * @param recordWriter Writes a single Join record to the DataOutputStream.
     * @param <F> Type of the first element in Join.
     * @param <S> Type of the second element in Join.
     */
    public static <F, S> void saveToIsam(
            String filePathBase,
            TupFrame<F, S> frame,
            IOConsumer<DataOutputStream, TupFrame<F, S>> metaWriter, // Changed to IOConsumer
            IOConsumer<DataOutputStream, Join<F, S>> recordWriter // Changed to IOConsumer
    ) throws IOException {

        // Write metadata
        try (DataOutputStream metaOut = new DataOutputStream(new FileOutputStream(filePathBase + ".bin.meta"))) {
            metaWriter.accept(metaOut, frame); // This call can now throw IOException
        }

        // Write data
        try (DataOutputStream dataOut = new DataOutputStream(new FileOutputStream(filePathBase + ".bin"))) {
            frame.forEach(join -> {
                try {
                    recordWriter.accept(dataOut, join);
                } catch (IOException e) {
                    // Wrap IOException in a RuntimeException because Stream.forEach's Consumer
                    // does not allow checked exceptions.
                    throw new RuntimeException("Error writing record to ISAM", e);
                }
            });
        }
    }

    // --- Conceptual ISAM Reading ---

    /**
     * Conceptually loads a TupFrame from a binary format (.bin) and its metadata file (.bin.meta).
     *
     * @param filePathBase The base name for the files.
     * @param metaReader Reads metadata and might configure the recordReader.
     * @param recordReader Reads a single Join record from the DataInputStream.
     * @param <F> Type of the first element in Join.
     * @param <S> Type of the second element in Join.
     * @return A TupFrame loaded with data.
     */
    public static <F, S> TupFrame<F, S> loadFromIsam(
            String filePathBase,
            IOFunction<DataInputStream, Object> metaReader, // Changed to IOFunction
            IOFunction<DataInputStream, Join<F, S>> recordReader // Changed to IOFunction
    ) throws IOException {
        // Read metadata (conceptual, not fully used by recordReader here for simplicity)
        try (DataInputStream metaIn = new DataInputStream(new FileInputStream(filePathBase + ".bin.meta"))) {
            /* Object metadata = */ metaReader.apply(metaIn); // This can now throw IOException
        }

        List<Join<F, S>> records = new ArrayList<>();
        try (DataInputStream dataIn = new DataInputStream(new FileInputStream(filePathBase + ".bin"))) {
            while (dataIn.available() > 0) { // Simple way to check for more data; might not be robust
                records.add(recordReader.apply(dataIn)); // This can now throw IOException
            }
        }
        return TupFrame.of(records);
    }
}
