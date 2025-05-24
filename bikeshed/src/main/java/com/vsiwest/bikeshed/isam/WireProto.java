package com.vsiwest.bikeshed.isam;

import com.example.bikeshed.dsel.RowVec;
import com.example.bikeshed.dsel.D;
import com.example.bikeshed.dsel.Series;
import com.example.bikeshed.types.ColumnMeta;
import com.example.bikeshed.types.IOMemento;

import java.nio.ByteBuffer;
import java.util.List;
import java.util.Objects;
import java.util.function.Function;

/**
 * Utility for reading and writing data records to/from a `ByteBuffer`
 * based on `RecordMeta` definitions. This is the core `bbcursive` integration point.
 */
public class WireProto {

    /**
     * Reads a `RowVec` from a `ByteBuffer` based on provided `RecordMeta` definitions.
     * This is the "decode" or "parsing" side, leveraging `bbcursive` principles.
     *
     * @param buffer The ByteBuffer containing the record's data.
     *               The buffer's position should be at the start of the record,
     *               and its limit should be at the end of the record.
     * @param recordMetas A list of `RecordMeta` defining the structure of the record.
     * @return A `RowVec` representing the parsed record.
     */
    public static RowVec readFromBuffer(ByteBuffer buffer, List<RecordMeta> recordMetas) {
        Objects.requireNonNull(buffer, "ByteBuffer cannot be null");
        Objects.requireNonNull(recordMetas, "RecordMetas cannot be null");

        // The RowVec provider takes a column index and returns a Join<Object, () -> ColumnMeta>
        return RowVec.of(recordMetas.size(), colIndex -> {
            RecordMeta colMeta = recordMetas.get(colIndex);
            int colBegin = colMeta.getBegin();
            int colEnd = colMeta.getEnd();
            IOMemento ioMemento = colMeta.getType(); // Get IOMemento from RecordMeta

            // Create a slice of the record buffer for the current column
            // This is crucial for zero-copy and direct bbcursive-like access.
            ByteBuffer columnSlice = buffer.duplicate();
            columnSlice.position(colBegin);
            columnSlice.limit(colEnd);
            ByteBuffer actualColumnBuffer = columnSlice.slice();

            // Use the IOMemento's decoder with the column's slice
            Object value = ioMemento.getDecoder().apply(actualColumnBuffer);

            // Return the Join<Value, LazyColumnMeta> for the RowVec
            return D.jn(value, (Function<Void, ColumnMeta>) unused -> colMeta);
        });
    }

    /**
     * Writes a `RowVec` to a `ByteBuffer` based on provided `RecordMeta` definitions.
     * This is the "encode" or "serialization" side.
     *
     * @param rowVec The `RowVec` containing the data to write.
     * @param buffer The `ByteBuffer` to write the data into.
     *               The buffer's position should be at the start of where the record needs to be written.
     * @param recordMetas A list of `RecordMeta` defining the structure of the record.
     * @return The `ByteBuffer` after the record has been written (position advanced).
     */
    public static ByteBuffer writeToBuffer(RowVec rowVec, ByteBuffer buffer, List<RecordMeta> recordMetas) {
        Objects.requireNonNull(rowVec, "RowVec cannot be null");
        Objects.requireNonNull(buffer, "ByteBuffer cannot be null");
        Objects.requireNonNull(recordMetas, "RecordMetas cannot be null");

        // Iterate through each column's metadata and its corresponding value in the RowVec
        for (int i = 0; i < recordMetas.size(); i++) {
            RecordMeta colMeta = recordMetas.get(i);
            Object colValue = rowVec.getValue(i); // Get the value for this column from the RowVec
            IOMemento ioMemento = colMeta.getType();

            // Get the target position within the record buffer for this column
            int targetPosition = colMeta.getBegin();

            // Create a temporary buffer for encoding if the IOMemento's encoder directly fills.
            // Or, the encoder accepts the target ByteBuffer and position.
            // Assuming IOMemento.getEncoder() expects (Object value, ByteBuffer targetBuffer)
            // It will handle writing to the correct position.
            ByteBuffer originalBufferState = buffer.duplicate(); // Save state
            originalBufferState.position(targetPosition);
            originalBufferState.limit(colMeta.getEnd()); // Limit to column boundary for safety

            // Use the IOMemento's encoder with the value and the sliced buffer
            ioMemento.getEncoder().accept(colValue, originalBufferState.slice());

            // Padding for variable-length strings if needed (handled by IO_STRING encoder itself)
            // The IO_STRING encoder within IOMemento is responsible for padding if its fixedSize is known.
            // The `writeToBuffer` only ensures the correct slice is given.
        }
        return buffer; // Return the buffer, its position might have been advanced depending on encoder impl
                       // For fixed-length records, it typically returns to the start of the next record.
    }
}
