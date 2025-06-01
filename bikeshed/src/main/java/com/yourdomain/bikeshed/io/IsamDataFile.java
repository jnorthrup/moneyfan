package com.yourdomain.bikeshed.io;

import borg.trikeshed.cursor.Cursor; // Changed
import borg.trikeshed.lib.Join;     // Changed
import borg.trikeshed.cursor.RowVec;   // Changed
import borg.trikeshed.lib.Series;    // Changed
import borg.trikeshed.isam.RecordMeta; // Changed
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.EnumSet;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static com.yourdomain.bikeshed.dsel.D.jn;

/**
 * Represents an ISAM (Indexed Sequential Access Method) data file.
 * This class provides methods to read data from a fixed-format binary file,
 * leveraging memory-mapped files and {@code bbcursive} for efficient, zero-copy access.
 * The schema is defined by an associated {@link IsamMetaFileReader}.
 */
public class IsamDataFile implements Cursor, AutoCloseable {

    private final String dataFileFilename;
    private final IsamMetaFileReader metaFile;
    private FileChannel fileChannel;
    private ByteBuffer mmapBuffer; // Memory-mapped buffer

    private int recordLengthBytes;
    private Series<IsamMetaFileReader.IsamColumnMeta> isamConstraints;
    private int recordCount;

    public IsamDataFile(@NotNull String dataFileFilename, @NotNull String metafileFilename) {
        this.dataFileFilename = dataFileFilename;
        this.metaFile = new IsamMetaFileReader(metafileFilename);
    }

    /**
     * Opens the ISAM data file, loads metadata, and memory-maps the file.
     * This method must be called before any data access operations.
     *
     * @throws IOException If the data file or metafile cannot be accessed.
     */
    public void open() throws IOException {
        metaFile.load();
        this.recordLengthBytes = metaFile.getRecordLengthBytes();
        this.isamConstraints = metaFile.getConstraints().alpha(cm -> {
            // Need to correctly cast or create IsamColumnMeta from ColumnMeta
            // This assumes the ColumnMeta loaded from metaFile is already an IsamColumnMeta
            // or can be converted. For now, direct cast as load() creates IsamColumnMeta.
            return (IsamMetaFileReader.IsamColumnMeta) cm;
        });

        Path dataPath = Paths.get(dataFileFilename);
        if (!Files.exists(dataPath)) {
            throw new IOException("Data file not found: " + dataFileFilename);
        }

        this.fileChannel = FileChannel.open(dataPath, StandardOpenOption.READ);
        long fileSize = fileChannel.size();
        if (fileSize % recordLengthBytes != 0) {
            System.err.println("Warning: Data file size (" + fileSize + " bytes) is not a multiple of record length (" + recordLengthBytes + " bytes). Data might be truncated or malformed.");
        }
        this.recordCount = (int) (fileSize / recordLengthBytes);

        // Memory-map the entire file for zero-copy access
        this.mmapBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, 0, fileSize);
    }

    /**
     * Closes the ISAM data file and releases the memory-mapped buffer.
     * @throws IOException If an I/O error occurs during closing.
     */
    @Override
    public void close() throws IOException {
        if (fileChannel != null) {
            fileChannel.close();
            fileChannel = null;
        }
        mmapBuffer = null; // Dereference the buffer
        recordCount = 0;
    }

    @Override
    public Integer fst() {
        return recordCount;
    }

    @Override
    public Function<Integer, RowVec> snd() {
        // The provider function for Cursor.get(index)
        return this::readRow;
    }

    /**
     * Reads a single row (record) from the memory-mapped buffer at the given index.
     * This uses {@code bbcursive} implicitly through the pre-configured decoders in {@code IsamColumnMeta}.
     *
     * @param rowIndex The 0-based index of the row to read.
     * @return A RowVec representing the data at that row.
     * @throws IndexOutOfBoundsException if the rowIndex is out of bounds.
     */
    private @NotNull RowVec readRow(int rowIndex) {
        if (rowIndex < 0 || rowIndex >= recordCount) {
            throw new IndexOutOfBoundsException("Row index " + rowIndex + " out of bounds for " + recordCount + " records.");
        }

        final int recordOffset = rowIndex * recordLengthBytes;

        // Create a slice of the mmapBuffer for this specific record.
        // This slice operation is "zero-copy" in the sense that it doesn't duplicate the underlying data.
        mmapBuffer.position(recordOffset);
        mmapBuffer.limit(recordOffset + recordLengthBytes);
        final ByteBuffer recordBuffer = mmapBuffer.slice(); // This slice will have position 0 and limit `recordLengthBytes`

        // Reset the main buffer's position and limit, as slice() and get() from slice() don't affect original buffer.
        mmapBuffer.position(0);
        mmapBuffer.limit(mmapBuffer.capacity());

        @SuppressWarnings("unchecked") // Suppress warning for creating generic array for toArray()
        Join<Object, Supplier<RecordMeta>>[] columnValuesAndMetaArray = IntStream.range(0, isamConstraints.size())
                .mapToObj(colIndex -> {
                    IsamMetaFileReader.IsamColumnMeta colMeta = isamConstraints.get(colIndex);
                    // Create a sub-slice for the column data within the recordBuffer
                    recordBuffer.position(colMeta.getBeginOffset());
                    recordBuffer.limit(colMeta.getEndOffset());
                    ByteBuffer columnDataBuffer = recordBuffer.slice();

                    // Decode the column value using the pre-configured decoder
                    Object value = colMeta.getDecoder().parse(columnDataBuffer).getValue(); // bbcursive parse

                    // Return a Join of the value and a supplier for its metadata
                    // IsamColumnMeta is a RecordMeta, so this cast is fine.
                    return jn(value, (Supplier<RecordMeta>) () -> colMeta);
                })
                .toArray(Join[]::new); // Collect to array

        return RowVec.of(columnValuesAndMetaArray); // Pass array to varargs method
    }

    /**
     * Writes a Cursor (tabular data) to an ISAM data file and its corresponding metadata file.
     * This operation is not memory-mapped for writing, but focuses on correctness for fixed-format.
     *
     * @param cursor The Cursor to write.
     * @param dataFilename The path for the data file.
     * @param varCharLengths A map of variable-length column names to their fixed lengths for ISAM.
     * @throws IOException If there's an error writing files.
     * @throws IllegalArgumentException If the Cursor is empty or schema cannot be determined.
     */
    public static void write(@NotNull Cursor cursor, @NotNull String dataFilename, @NotNull Map<String, Integer> varCharLengths) throws IOException {
        if (cursor.size() == 0) {
            throw new IllegalArgumentException("Cannot write an empty Cursor to ISAM file.");
        }

        // 1. Determine the schema (ColumnMeta) and write the metadata file
        Series<RecordMeta> schema = cursor.meta(); // cursor.meta() now returns Series<RecordMeta>
        String metaFilename = dataFilename + ".meta";
        IsamMetaFileReader.write(metaFilename, schema, varCharLengths); // IsamMetaFileReader.write now expects Series<RecordMeta>

        // Re-load the metafile to get the concrete IsamColumnMeta with offsets, decoders, and encoders.
        IsamMetaFileReader writerMetaReader = new IsamMetaFileReader(metaFilename);
        writerMetaReader.load();
        Series<IsamMetaFileReader.IsamColumnMeta> writeConstraints = writerMetaReader.getConstraints().alpha(cm -> (IsamMetaFileReader.IsamColumnMeta) cm);
        int recordLength = writerMetaReader.getRecordLengthBytes();

        // 2. Write the data file
        Path dataPath = Paths.get(dataFilename);
        try (FileChannel channel = FileChannel.open(dataPath, EnumSet.of(StandardOpenOption.CREATE, StandardOpenOption.WRITE, StandardOpenOption.TRUNCATE_EXISTING))) {
            for (int i = 0; i < cursor.size(); i++) {
                RowVec row = cursor.get(i);
                ByteBuffer recordBuffer = ByteBuffer.allocate(recordLength); // Buffer for the entire row

                for (int j = 0; j < writeConstraints.size(); j++) {
                    IsamMetaFileReader.IsamColumnMeta colMeta = writeConstraints.get(j);
                    Object value = row.get(j).fst(); // Get the raw value from the RowVec

                    // Encode the value using the pre-configured encoder
                    ByteBuffer encodedColumn = colMeta.getEncoder().apply(value);

                    // Put the encoded bytes into the record buffer at the correct offset
                    recordBuffer.position(colMeta.getBeginOffset());
                    recordBuffer.put(encodedColumn); // Puts remaining bytes from encodedColumn
                }
                recordBuffer.flip(); // Prepare recordBuffer for writing to channel
                channel.write(recordBuffer);
            }
        }
    }

    /**
     * Appends additional rows to an existing ISAM data file.
     * Assumes the data file and metafile already exist and are consistent.
     *
     * @param rowsToAppend The iterable of RowVecs to append.
     * @param dataFilename The path to the data file.
     * @param varCharLengths A map of variable-length column names to their fixed lengths for ISAM.
     *                       Required to correctly format new rows, even if file exists.
     * @param transform An optional function to transform each RowVec before appending.
     * @throws IOException If there's an error appending to the file.
     * @throws IllegalArgumentException If the schema cannot be determined or is inconsistent.
     */
    public static void append(@NotNull Iterable<RowVec> rowsToAppend, @NotNull String dataFilename, @NotNull Map<String, Integer> varCharLengths, @Nullable Function<RowVec, RowVec> transform) throws IOException {
        String metaFilename = dataFilename + ".meta";
        Path dataPath = Paths.get(dataFilename);

        if (!Files.exists(dataPath) || !Files.exists(Paths.get(metaFilename))) {
            throw new IllegalArgumentException("Cannot append: Data file or metafile does not exist. Use write() for initial creation.");
        }

        // Load existing metadata to get the schema for appending
        IsamMetaFileReader appendMetaReader = new IsamMetaFileReader(metaFilename);
        appendMetaReader.load();
        Series<IsamMetaFileReader.IsamColumnMeta> appendConstraints = appendMetaReader.getConstraints().alpha(cm -> (IsamMetaFileReader.IsamColumnMeta) cm);
        int recordLength = appendMetaReader.getRecordLengthBytes();

        try (FileChannel channel = FileChannel.open(dataPath, EnumSet.of(StandardOpenOption.APPEND))) {
            for (RowVec originalRow : rowsToAppend) {
                RowVec row = (transform != null) ? transform.apply(originalRow) : originalRow;
                ByteBuffer recordBuffer = ByteBuffer.allocate(recordLength);

                for (int j = 0; j < appendConstraints.size(); j++) {
                    IsamMetaFileReader.IsamColumnMeta colMeta = appendConstraints.get(j);
                    Object value = row.get(j).fst();

                    ByteBuffer encodedColumn = colMeta.getEncoder().apply(value);
                    recordBuffer.position(colMeta.getBeginOffset());
                    recordBuffer.put(encodedColumn);
                }
                recordBuffer.flip();
                channel.write(recordBuffer);
            }
        }
    }
}
