package com.yourdomain.bikeshed.io;

import com.yourdomain.bikeshed.core.Series;
import com.yourdomain.bikeshed.type.ColumnMeta;
import com.yourdomain.bbcursive.core.Cursive;
import org.jetbrains.annotations.NotNull;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * Reads ISAM metadata files to define the schema for data files.
 * The metafile specifies column names, types, and their byte offsets within a record.
 * This ensures strict, fixed-format data access suitable for memory-mapped files.
 */
public class IsamMetaFileReader {

    private final String metafileFilename;
    private Series<ColumnMeta> constraints;
    private int recordLengthBytes;

    public IsamMetaFileReader(@NotNull String metafileFilename) {
        this.metafileFilename = metafileFilename;
    }

    /**
     * Loads the metadata from the file and populates the constraints.
     * This method assumes the metadata file format as described in the prompt.
     *
     * @throws IOException If the metafile cannot be read.
     * @throws IllegalArgumentException If the metafile is malformed.
     */
    public void load() throws IOException {
        Path path = Paths.get(metafileFilename);
        if (!Files.exists(path)) {
            throw new IOException("Metafile not found: " + metafileFilename);
        }

        List<String> lines = Files.readAllLines(path);
        List<String> dataLines = lines.stream()
                .filter(line -> !line.trim().startsWith("#") && !line.isBlank())
                .collect(Collectors.toList());

        if (dataLines.size() < 3) {
            throw new IllegalArgumentException("Metafile is too short, expected at least 3 data lines.");
        }

        String[] coordStrings = dataLines.get(0).trim().split("\\s+");
        String[] nameStrings = dataLines.get(1).trim().split("\\s+");
        String[] typeStrings = dataLines.get(2).trim().split("\\s+");

        if (nameStrings.length != typeStrings.length || coordStrings.length != nameStrings.length * 2) {
            throw new IllegalArgumentException("Malformed metafile: mismatch in counts of coords, names, or types.");
        }

        List<ColumnMeta> tempConstraints = new ArrayList<>();
        for (int i = 0; i < nameStrings.length; i++) {
            String name = nameStrings[i];
            IOMemento type;
            int fixedSize;

            // Handle FixedSizeTypeMemento (e.g., IoString(10))
            if (typeStrings[i].contains("(")) {
                String baseTypeName = typeStrings[i].substring(0, typeStrings[i].indexOf("("));
                type = IOMemento.valueOf(baseTypeName);
                fixedSize = Integer.parseInt(typeStrings[i].substring(typeStrings[i].indexOf("(") + 1, typeStrings[i].indexOf(")")));
                if (type.networkSize() != null) { // If base type has a fixed size, ensure it matches
                    if (type.networkSize() != fixedSize) {
                        throw new IllegalArgumentException("Mismatch in fixed size for " + name + ": Declared " + fixedSize + ", but base type " + type.name() + " has " + type.networkSize());
                    }
                }
            } else {
                type = IOMemento.valueOf(typeStrings[i]);
                fixedSize = type.networkSize() != null ? type.networkSize() : -1; // -1 means variable, but should be fixed by coords
            }

            int fieldBegin = Integer.parseInt(coordStrings[2 * i]);
            int fieldEnd = Integer.parseInt(coordStrings[2 * i + 1]);
            int actualFieldLength = fieldEnd - fieldBegin;

            if (fixedSize != -1 && actualFieldLength != fixedSize) {
                throw new IllegalArgumentException("Mismatch in fixed size for " + name + ": Declared " + fixedSize + ", but actual length from coords is " + actualFieldLength);
            }
            if (actualFieldLength <= 0) {
                throw new IllegalArgumentException("Invalid field length for " + name + ": " + actualFieldLength);
            }

            tempConstraints.add(new IsamColumnMeta(name, type, fieldBegin, fieldEnd));
        }
        this.constraints = Series.of(tempConstraints.size(), tempConstraints::get);
        this.recordLengthBytes = Integer.parseInt(coordStrings[coordStrings.length - 1]); // Last coord is total record length

        // Verify calculated record length matches the one in metadata
        int calculatedRecordLength = tempConstraints.stream()
                .mapToInt(IsamColumnMeta::getLength)
                .sum();
        if (calculatedRecordLength != recordLengthBytes) {
            throw new IllegalArgumentException("Calculated record length (" + calculatedRecordLength + ") does not match metafile record length (" + recordLengthBytes + ").");
        }
    }

    /**
     * Retrieves the loaded column metadata constraints.
     * Must call {@code load()} first.
     * @return A Series of ColumnMeta representing the schema.
     */
    public @NotNull Series<ColumnMeta> getConstraints() {
        if (constraints == null) {
            throw new IllegalStateException("Metadata not loaded. Call load() first.");
        }
        return constraints;
    }

    /**
     * Returns the fixed length of each record in bytes.
     * Must call {@code load()} first.
     * @return The record length in bytes.
     */
    public int getRecordLengthBytes() {
        if (constraints == null) {
            throw new IllegalStateException("Metadata not loaded. Call load() first.");
        }
        return recordLengthBytes;
    }

    /**
     * Creates an ISAM metafile from a Series of ColumnMeta.
     * Automatically calculates offsets and lengths for fixed-size types.
     * For variable-length types (like String, ByteArray), their lengths must be provided
     * in the `varCharLengths` map.
     *
     * @param metafileFilename The path to write the metafile.
     * @param columnMetas A Series of ColumnMeta defining the schema.
     * @param varCharLengths A map of variable-length column names to their fixed lengths in bytes.
     * @throws IOException If the file cannot be written.
     * @throws IllegalArgumentException If variable-length types don't have lengths specified.
     */
    public static void write(@NotNull String metafileFilename, @NotNull Series<ColumnMeta> columnMetas, @NotNull Map<String, Integer> varCharLengths) throws IOException {
        List<String> coordStrings = new ArrayList<>();
        List<String> nameStrings = new ArrayList<>();
        List<String> typeStrings = new ArrayList<>();

        int currentOffset = 0;
        for (int i = 0; i < columnMetas.size(); i++) {
            ColumnMeta colMeta = columnMetas.get(i);
            String name = colMeta.name();
            TypeMemento typeMemento = colMeta.type(); // Can be IOMemento or FixedSizeTypeMemento

            nameStrings.add(name);

            int fieldLength;
            if (typeMemento.networkSize() != null) {
                fieldLength = typeMemento.networkSize();
                typeStrings.add(typeMemento instanceof IOMemento ? ((IOMemento) typeMemento).name() : typeMemento.toString());
            } else {
                fieldLength = varCharLengths.getOrDefault(name, -1);
                if (fieldLength == -1) {
                    throw new IllegalArgumentException("Variable-length type '" + name + "' (Type: " + typeMemento.getClass().getSimpleName() + ") requires an explicit length in varCharLengths map.");
                }
                typeStrings.add(typeMemento.toString()); // e.g., IoString(10)
            }

            coordStrings.add(String.valueOf(currentOffset));
            currentOffset += fieldLength;
            coordStrings.add(String.valueOf(currentOffset));
        }
        coordStrings.add(String.valueOf(currentOffset)); // Add total record length

        List<String> lines = new ArrayList<>();
        lines.add("# format:  coords WS .. EOL names WS .. EOL TypeMememento WS ..");
        lines.add("# last coord is the recordlen");
        lines.add(String.join(" ", coordStrings));
        lines.add(String.join(" ", nameStrings));
        lines.add(String.join(" ", typeStrings));

        Files.write(Paths.get(metafileFilename), lines);
    }

    // A specialized ColumnMeta for ISAM to hold begin/end/decoder/encoder
    // In a real system, this would be a separate record or a richer ColumnMeta interface.
    // For this example, we define it as a concrete class to hold the necessary details.
    public static class IsamColumnMeta extends ColumnMeta.ImmutableColumnMeta {
        private final int beginOffset;
        private final int endOffset;
        private final Cursive<Object> decoder;
        private final Function<Object, ByteBuffer> encoder;

        public IsamColumnMeta(@NotNull String name, @NotNull IOMemento type, int beginOffset, int endOffset) {
            super(name, type);
            this.beginOffset = beginOffset;
            this.endOffset = endOffset;
            int length = endOffset - beginOffset;
            this.decoder = type.createDecoder(length);
            this.encoder = type.createEncoder(length);
        }

        // Constructor for FixedSizeTypeMemento
        public IsamColumnMeta(@NotNull String name, @NotNull D.FixedSizeTypeMemento type, int beginOffset, int endOffset) {
            super(name, type);
            this.beginOffset = beginOffset;
            this.endOffset = endOffset;
            int length = endOffset - beginOffset;
            this.decoder = type.getBaseType().createDecoder(length);
            this.encoder = type.getBaseType().createEncoder(length);
        }

        public int getBeginOffset() {
            return beginOffset;
        }

        public int getEndOffset() {
            return endOffset;
        }

        public @NotNull Cursive<Object> getDecoder() {
            return decoder;
        }

        public @NotNull Function<Object, ByteBuffer> getEncoder() {
            return encoder;
        }

        public int getLength() {
            return endOffset - beginOffset;
        }

        @Override
        public String toString() {
            return "ISAMColumnMeta{" +
                   "name='" + name() + '\'' +
                   ", type=" + type() +
                   ", begin=" + beginOffset +
                   ", end=" + endOffset +
                   ", length=" + getLength() +
                   '}';
        }
    }
}
