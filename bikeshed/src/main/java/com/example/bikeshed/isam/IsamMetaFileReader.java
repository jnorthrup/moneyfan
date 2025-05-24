package com.example.bikeshed.isam;

import com.example.bikeshed.dsel.D;
import com.example.bikeshed.dsel.Series;
import com.example.bikeshed.types.ColumnMeta;
import com.example.bikeshed.types.IOMemento;
import com.example.bikeshed.types.TypeMemento;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.stream.Collectors;

/**
 * Reads and writes metadata files for ISAM data files.
 * The metadata defines column structure, types, and byte offsets within a record.
 *
 * Format:
 * # format:  coords WS .. EOL names WS .. EOL TypeMememento WS ..
 * # last coord is the recordlen
 * 0 12 12 24 24 32 ... (coords: begin_0 end_0 begin_1 end_1 ...)
 * Open_time Close_time Open High Low Close Volume ... (names)
 * IoInstant IoInstant IoDouble IoDouble IoDouble IoDouble ... (types)
 */
public class IsamMetaFileReader implements AutoCloseable {

    private final String metafileFilename;
    private List<RecordMeta> constraints;
    private int recordLength = 0;

    public IsamMetaFileReader(String metafileFilename) {
        this.metafileFilename = Objects.requireNonNull(metafileFilename);
    }

    /**
     * Opens the metadata file and parses its contents.
     * This method initializes the `constraints` and `recordLength`.
     *
     * @throws IOException If the file cannot be read or is malformed.
     */
    public void open() throws IOException {
        if (constraints != null && !constraints.isEmpty()) {
            return; // Already opened
        }

        Path metaFilePath = Path.of(metafileFilename);
        if (!Files.exists(metaFilePath)) {
            throw new IOException("Metafile does not exist: " + metafileFilename);
        }

        List<String> lines = Files.readAllLines(metaFilePath);
        List<String> dataLines = lines.stream()
                .filter(line -> !line.trim().startsWith("#") && !line.isBlank())
                .collect(Collectors.toList());

        if (dataLines.size() < 3) {
            throw new IOException("Metafile is malformed, expected at least 3 data lines (coords, names, types).");
        }

        String[] coordsStr = dataLines.get(0).trim().split("\\s+");
        String[] names = dataLines.get(1).trim().split("\\s+");
        String[] typesStr = dataLines.get(2).trim().split("\\s+");

        if (names.length != typesStr.length || coordsStr.length != names.length * 2) {
            throw new IOException("Metafile mismatch: column count mismatch between coords, names, and types.");
        }

        this.constraints = new ArrayList<>(names.length);
        for (int i = 0; i < names.length; i++) {
            String name = names[i];
            IOMemento ioMemento = IOMemento.fromTypeName(typesStr[i]);
            int begin = Integer.parseInt(coordsStr[2 * i]);
            int end = Integer.parseInt(coordsStr[2 * i + 1]);

            // Create RecordMeta with appropriate decoder/encoder based on IOMemento and actual size
            RecordMeta recordMeta = new RecordMeta(
                    name,
                    ioMemento,
                    begin,
                    end,
                    ioMemento.getDecoder(), // Decoders might need ByteBuffer configured for size
                    ioMemento.getEncoder()  // Encoders might need ByteBuffer configured for size
            );
            this.constraints.add(recordMeta);
        }

        // The last 'end' coordinate is the total record length.
        if (!this.constraints.isEmpty()) {
            this.recordLength = this.constraints.get(this.constraints.size() - 1).getEnd();
        }
    }

    @Override
    public void close() {
        // No-op for file readers as they don't hold open channels in this design
        // unless you explicitly manage `FileChannel` objects here.
        // For simplicity, assuming file is read entirely in `open()`.
    }

    public List<RecordMeta> getConstraints() {
        if (constraints == null) {
            throw new IllegalStateException("IsamMetaFileReader not opened yet. Call open() first.");
        }
        return constraints;
    }

    public int getRecordLength() {
        if (recordLength == 0 && (constraints == null || constraints.isEmpty())) {
            throw new IllegalStateException("IsamMetaFileReader not opened yet or has no constraints.");
        }
        return recordLength;
    }

    /**
     * Writes metadata to a file.
     * This method sanitizes `ColumnMeta` to `RecordMeta` by assigning byte offsets
     * and ensuring fixed sizes for variable-length types if specified.
     *
     * @param metafilename The path to the metadata file.
     * @param columnMetas A Series of `ColumnMeta` objects to write.
     * @param varChars A map from column name to its fixed byte length for variable-length types.
     * @return A Series of `RecordMeta` objects (the sanitized version).
     * @throws IOException If the file cannot be written.
     * @throws IllegalArgumentException If variable-length types don't have a specified length.
     */
    public static Series<ColumnMeta> write(String metafilename, Series<ColumnMeta> columnMetas, Map<String, Integer> varChars) throws IOException {
        List<String> coords = new ArrayList<>();
        List<String> names = new ArrayList<>();
        List<String> types = new ArrayList<>();
        List<RecordMeta> sanitizedMetas = new ArrayList<>();

        int currentOffset = 0;
        for (int i = 0; i < columnMetas.size(); i++) {
            ColumnMeta colMeta = columnMetas.get(i);
            String name = colMeta.getName();
            TypeMemento type = colMeta.getType();

            int length;
            if (type.getNetworkSize() != null) {
                length = type.getNetworkSize();
            } else {
                // Variable-length type, look up its configured length
                if (!varChars.containsKey(name)) {
                    throw new IllegalArgumentException("Variable-length type '" + name + "' (" + type.getClass().getSimpleName() + ") requires explicit 'networkSize' in varChars map.");
                }
                length = varChars.get(name);
            }

            int begin = currentOffset;
            int end = currentOffset + length;

            // Create RecordMeta with the correct byte offsets and IOMemento
            RecordMeta recordMeta = new RecordMeta(
                    name,
                    (IOMemento) type, // Cast to IOMemento, assuming all TypeMemento in DSEL are IOMemento
                    begin,
                    end,
                    ((IOMemento)type).getDecoder(), // Pass the decoder from IOMemento
                    ((IOMemento)type).getEncoder()  // Pass the encoder from IOMemento
            );
            sanitizedMetas.add(recordMeta);

            coords.add(String.valueOf(begin));
            coords.add(String.valueOf(end));
            names.add(name);
            types.add(type.getClass().getSimpleName().toUpperCase()); // Use enum name, e.g., IO_STRING

            currentOffset += length;
        }

        // Write to file
        List<String> lines = new ArrayList<>();
        lines.add("# format: coords WS .. EOL names WS .. EOL TypeMememento WS ..");
        lines.add("# last coord is the recordlen");
        lines.add(String.join(" ", coords));
        lines.add(String.join(" ", names));
        lines.add(String.join(" ", types));

        Files.write(Path.of(metafilename), lines, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);

        return D.sr(sanitizedMetas.size(), sanitizedMetas::get);
    }
}
