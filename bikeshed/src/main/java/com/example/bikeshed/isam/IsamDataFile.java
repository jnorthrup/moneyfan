package com.example.bikeshed.isam;

import com.example.bikeshed.bbcursive.Cursive;
import com.example.bikeshed.bbcursive.util.ByteParsers;
import com.example.bikeshed.dsel.Cursor;
import com.example.bikeshed.dsel.D;
import com.example.bikeshed.dsel.Join;
import com.example.bikeshed.dsel.RowVec;
import com.example.bikeshed.types.ColumnMeta;
import com.example.bikeshed.types.IOMemento;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Map;
import java.util.Objects;
import java.util.function.IntFunction;
import java.util.function.Function;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Handles structured, fixed-format file I/O for ISAM data files.
 * Directly leverages `bbcursive` for byte-level parsing and memory-mapped files (mmap)
 * for zero-copy operations.
 *
 * Implements `Cursor` for seamless DSEL integration.
 */
public class IsamDataFile extends Cursor implements AutoCloseable {

    private final String datafileFilename;
    private final String metafileFilename;
    private final IsamMetaFileReader metafileReader;

    private FileChannel fileChannel;
    private ByteBuffer mmapBuffer;
    private int recordLength; // Derived from metadata

    // Private constructor, used by the static factory method
    private IsamDataFile(String datafileFilename, String metafileFilename, IsamMetaFileReader metafileReader,
                         int numRecords, IntFunction<RowVec> rowProvider) {
        super(numRecords, rowProvider);
        this.datafileFilename = Objects.requireNonNull(datafileFilename);
        this.metafileFilename = Objects.requireNonNull(metafileFilename);
        this.metafileReader = Objects.requireNonNull(metafileReader);
    }

    /**
     * Factory method to create an IsamDataFile instance (which is a Cursor).
     * This method handles opening the metafile and data file, and memory-mapping.
     *
     * @param datafileFilename Path to the data file.
     * @param metafileFilename Path to the metadata file.
     * @return A new IsamDataFile instance (as a Cursor).
     * @throws IOException If files cannot be accessed.
     */
    public static IsamDataFile create(String datafileFilename, String metafileFilename) throws IOException {
        IsamMetaFileReader metaReader = new IsamMetaFileReader(metafileFilename);
        metaReader.open(); // Load metadata eagerly to get recordLength

        Path dataFilePath = Path.of(datafileFilename);
        if (!Files.exists(dataFilePath)) {
            throw new IOException("Data file does not exist for reading: " + datafileFilename);
        }

        FileChannel channel = FileChannel.open(dataFilePath, StandardOpenOption.READ);
        long fileSize = channel.size();
        int recordLen = metaReader.getRecordLength();
        int numRecords = (int) (fileSize / recordLen);

        ByteBuffer mappedBuffer = channel.map(FileChannel.MapMode.READ_ONLY, 0, fileSize);

        // This is the core 'provider' for the Cursor/Series
        IntFunction<RowVec> rowProvider = recordIndex -> {
            int recordOffset = recordIndex * recordLen;
            // Create a slice for the current record's bytes
            ByteBuffer recordBuffer = mappedBuffer.duplicate();
            recordBuffer.position(recordOffset);
            recordBuffer.limit(recordOffset + recordLen);
            ByteBuffer currentRecordSlice = recordBuffer.slice(); // This is the buffer for one record

            return WireProto.readFromBuffer(currentRecordSlice, metaReader.getConstraints());
        };

        IsamDataFile isamDataFile = new IsamDataFile(datafileFilename, metafileFilename, metaReader, numRecords, rowProvider);
        isamDataFile.fileChannel = channel;
        isamDataFile.mmapBuffer = mappedBuffer;
        isamDataFile.recordLength = recordLen;

        return isamDataFile;
    }

    @Override
    public void close() throws IOException {
        if (fileChannel != null) {
            fileChannel.close();
            fileChannel = null;
            mmapBuffer = null;
        }
        metafileReader.close();
    }

    /**
     * Writes a `Cursor` to an ISAM data file and creates its metadata file.
     * @param cursor The Cursor containing data to write.
     * @param datafilename The path to the data file.
     * @param varChars A map specifying fixed lengths for variable-length strings (columnName -> length).
     * @throws IOException If file operations fail.
     */
    public static void write(Cursor cursor, String datafilename, Map<String, Integer> varChars) throws IOException {
        // First, derive or sanitize metadata and write the meta file
        List<ColumnMeta> rawMetas = cursor.getMetaData().stream().collect(Collectors.toList());
        com.example.bikeshed.dsel.Series<ColumnMeta> sanitizedMetas = IsamMetaFileReader.write(
                datafilename + ".meta", D.sr(rawMetas.size(), rawMetas::get), varChars);

        int recordLen = ((RecordMeta) sanitizedMetas.get(0)).getEnd(); // Get total record length from first meta

        Path dataFilePath = Path.of(datafilename);
        try (FileChannel fileChannel = FileChannel.open(dataFilePath, StandardOpenOption.CREATE, StandardOpenOption.WRITE, StandardOpenOption.TRUNCATE_EXISTING)) {
            ByteBuffer buffer = ByteBuffer.allocateDirect(recordLen); // Allocate direct buffer for performance

            for (RowVec row : D.iterable(cursor)) {
                buffer.clear(); // Reset buffer for new record
                WireProto.writeToBuffer(row, buffer, sanitizedMetas.stream().map(m -> (RecordMeta) m).collect(Collectors.toList()));
                buffer.flip(); // Prepare for writing
                while (buffer.hasRemaining()) {
                    fileChannel.write(buffer);
                }
            }
        }
    }

    /**
     * Appends `RowVec`s to an existing ISAM data file.
     * @param rowsToAppend An iterable of RowVecs to append.
     * @param datafilename The path to the data file.
     * @param varChars A map specifying fixed lengths for variable-length strings (columnName -> length).
     *                       Required to correctly format new rows, even if file exists.
     * @param transform An optional function to transform each RowVec before appending.
     * @throws IOException If there's an error appending to the file.
     * @throws IllegalArgumentException If the schema cannot be determined or is inconsistent.
     */
    public static void append(
            Iterable<RowVec> rowsToAppend,
            String datafilename,
            Map<String, Integer> varChars,
            Function<RowVec, RowVec> transform) throws IOException {

        String metafileFilename = datafilename + ".meta";
        Path dataFilePath = Path.of(datafilename);

        if (!Files.exists(dataFilePath) || !Files.exists(Path.of(metafileFilename))) {
            throw new IllegalArgumentException("Cannot append: Data file or metafile does not exist. Use write() for initial creation.");
        }

        // Load existing metadata to get the schema for appending
        IsamMetaFileReader appendMetaReader = new IsamMetaFileReader(metafileFilename);
        appendMetaReader.open(); // Load existing metadata
        com.example.bikeshed.dsel.Series<ColumnMeta> existingMetas = D.sr(appendMetaReader.getConstraints().size(), appendMetaReader.getConstraints()::get);
        int recordLen = ((RecordMeta) existingMetas.get(0)).getEnd();

        try (FileChannel fileChannel = FileChannel.open(dataFilePath, StandardOpenOption.APPEND, StandardOpenOption.WRITE)) {
            ByteBuffer buffer = ByteBuffer.allocateDirect(recordLen);

            for (RowVec originalRow : rowsToAppend) {
                RowVec row = (transform != null) ? transform.apply(originalRow) : originalRow;
                buffer.clear();
                WireProto.writeToBuffer(row, buffer, existingMetas.stream().map(m -> (RecordMeta)m).collect(Collectors.toList()));
                buffer.flip();
                while (buffer.hasRemaining()) {
                    fileChannel.write(buffer);
                }
            }
        }
    }
}
