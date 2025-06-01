package com.yourdomain.bikeshed.io;

import borg.trikeshed.nio.IOMemento; // Changed from isam.meta to nio
import borg.trikeshed.isam.RecordMeta;   // Changed
import borg.trikeshed.lib.Series; // Changed to new location
import com.yourdomain.bbcursive.core.Cursive;
import com.yourdomain.bikeshed.dsel.D; // For FixedSizeTypeMemento
import org.jetbrains.annotations.NotNull;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List; // Still needed for Files.readAllLines
import java.util.Map;
import java.util.function.Function;
// import java.util.stream.Collectors; // No longer needed for dataLines
import java.util.stream.IntStream; // For loop if used, or can use standard for loop

/**
 * Reads ISAM metadata files to define the schema for data files.
 * The metafile specifies column names, types, and their byte offsets within a record.
 * This ensures strict, fixed-format data access suitable for memory-mapped files.
 */
public class IsamMetaFileReader {

    private final String metafileFilename;
    private Series<RecordMeta> constraints; // Changed
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

        List<String> linesList = Files.readAllLines(path); // Step 1: Keep as is
        Series<String> linesSeries = Series.of(linesList.size(), linesList::get); // Step 1: Convert to Series

        // Step 2: Filter lines using Series.filter
        Series<String> filteredLinesSeries = linesSeries.filter(line -> !line.trim().startsWith("#") && !line.isBlank());

        if (filteredLinesSeries.size() < 3) { // Step 3: Check size
            throw new IllegalArgumentException("Metafile is too short, expected at least 3 data lines.");
        }

        // Step 3: Access specific definition lines
        String coordStringsLine = filteredLinesSeries.get(0).trim();
        String nameStringsLine = filteredLinesSeries.get(1).trim();
        String typeStringsLine = filteredLinesSeries.get(2).trim();

        String[] coordStrings = coordStringsLine.split("\\s+");
        String[] nameStrings = nameStringsLine.split("\\s+");
        String[] typeStrings = typeStringsLine.split("\\s+");

        if (nameStrings.length != typeStrings.length || coordStrings.length != nameStrings.length * 2) {
            throw new IllegalArgumentException("Malformed metafile: mismatch in counts of coords, names, or types.");
        }

        // Step 4: Parse and collect RecordMeta objects into an array
        RecordMeta[] constraintsArray = new RecordMeta[nameStrings.length];
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

            // Handle potential creation of D.FixedSizeTypeMemento if type is IoString or IoByteArray and fixedSize is from typeStrings[i]
            if (typeStrings[i].contains("(")) {
                // This logic was already present and correctly determines 'type' (base enum) and 'fixedSize'
                // We need to pass the correct IOMemento instance to IsamColumnMeta constructor
                // If it's a fixed-size spec for a variable type like IoString(10), we might need FixedSizeTypeMemento
                // However, IsamColumnMeta's first constructor takes IOMemento (enum), second takes D.FixedSizeTypeMemento
                // The current logic correctly deduces the base 'type' (enum) and 'fixedSize'.
                // The IsamColumnMeta constructor that takes the base enum 'type' will use 'actualFieldLength' for its decoder/encoder.
                // This seems fine. If a FixedSizeTypeMemento was intended, the caller of IsamColumnMeta would create it.
                // Here, we are parsing the meta file, so we create IsamColumnMeta with the base enum type.
                constraintsArray[i] = new IsamColumnMeta(name, type, fieldBegin, fieldEnd);
            } else {
                 constraintsArray[i] = new IsamColumnMeta(name, type, fieldBegin, fieldEnd);
            }
        }
        // Step 5: Assign to this.constraints
        this.constraints = Series.of(constraintsArray.length, idx -> constraintsArray[idx]);
        this.recordLengthBytes = Integer.parseInt(coordStrings[coordStrings.length - 1]); // Last coord is total record length

        // Verify calculated record length matches the one in metadata
        // Need to iterate over constraintsArray or this.constraints Series
        int calculatedRecordLength = 0;
        for(int i=0; i < constraintsArray.length; ++i) {
            calculatedRecordLength += ((IsamColumnMeta)constraintsArray[i]).getLength();
        }
        // Or using the new Series:
        // int calculatedRecordLength = IntStream.range(0, this.constraints.size())
        //        .map(i -> ((IsamColumnMeta)this.constraints.get(i)).getLength())
        //        .sum();

        if (calculatedRecordLength != recordLengthBytes) {
            throw new IllegalArgumentException("Calculated record length (" + calculatedRecordLength + ") does not match metafile record length (" + recordLengthBytes + ").");
        }
    }

    /**
     * Retrieves the loaded column metadata constraints.
     * Must call {@code load()} first.
     * @return A Series of ColumnMeta representing the schema.
     */
    public @NotNull Series<RecordMeta> getConstraints() { // Changed
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
    public static void write(@NotNull String metafileFilename, @NotNull Series<RecordMeta> columnMetas, @NotNull Map<String, Integer> varCharLengths) throws IOException { // Changed
        List<String> coordStrings = new ArrayList<>();
        List<String> nameStrings = new ArrayList<>();
        List<String> typeStrings = new ArrayList<>();

        int currentOffset = 0;
        for (int i = 0; i < columnMetas.size(); i++) {
            RecordMeta colMeta = columnMetas.get(i); // Changed
            String name = colMeta.name();
            IOMemento typeMemento = colMeta.type(); // colMeta.type() now returns IOMemento enum

            nameStrings.add(name);

            int fieldLength;
            // typeMemento is now the enum. It could have been wrapped in FixedSizeTypeMemento by user.
            // However, RecordMeta stores the base enum type.
            // The .toString() of FixedSizeTypeMemento is "IoString(10)".
            // The .name() of IOMemento enum is "IoString".
            // The meta file expects "IoString(10)" for fixed variable types.
            // This means if colMeta came from D.fsString(10), its type() would be IoString enum.
            // This static write method needs to know the intended fixed length for variable types.
            // This implies columnMetas should perhaps be Series<IsamColumnMeta> or similar rich type.
            // Or, the ColumnMeta/RecordMeta itself should store the "fixed variable length" if specified at creation.
            // Current RecordMeta does not store this. It only stores the base IOMemento enum.
            // Let's assume varCharLengths map is the source of truth for fixed variable lengths.
            if (typeMemento.networkSize() != null) { // Truly fixed size type (e.g. IoInt)
                fieldLength = typeMemento.networkSize();
                typeStrings.add(typeMemento.name()); // Use enum name: "IoInt"
            } else { // Variable size type (e.g. IoString, IoByteArray)
                fieldLength = varCharLengths.getOrDefault(name, -1);
                if (fieldLength == -1) {
                    throw new IllegalArgumentException("Variable-length type '" + name + "' (Type: " + typeMemento.name() + ") requires an explicit length in varCharLengths map.");
                }
                typeStrings.add(typeMemento.name() + "(" + fieldLength + ")"); // e.g., "IoString(10)"
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
    public static class IsamColumnMeta extends RecordMeta.ImmutableRecordMeta { // Changed superclass
        private final int beginOffset;
        private final int endOffset;
        private final Cursive<Object> decoder;
        private final Function<Object, ByteBuffer> encoder;

        public IsamColumnMeta(@NotNull String name, @NotNull IOMemento type, int beginOffset, int endOffset) { // IOMemento is now enum
            super(name, type); // Super constructor expects (String, IOMemento enum)
            this.beginOffset = beginOffset;
            this.endOffset = endOffset;
            int length = endOffset - beginOffset;
            this.decoder = type.createDecoder(length);
            this.encoder = type.createEncoder(length);
        }

        // Constructor for FixedSizeTypeMemento
        public IsamColumnMeta(@NotNull String name, @NotNull D.FixedSizeTypeMemento type, int beginOffset, int endOffset) { // type is D.FixedSizeTypeMemento
            super(name, type); // Super constructor expects (String, IOMemento). D.FixedSizeTypeMemento implements IOMemento.
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
