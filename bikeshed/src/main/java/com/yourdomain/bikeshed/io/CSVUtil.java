package com.yourdomain.bikeshed.io;

import borg.trikeshed.cursor.Cursor;   // Changed
import borg.trikeshed.lib.Join;     // Changed (via D.jn)
import borg.trikeshed.cursor.RowVec;   // Changed (via D.rv)
import borg.trikeshed.lib.Series; // Added for Series usage
import com.yourdomain.bikeshed.dsel.D;     // D itself is updated
import borg.trikeshed.isam.RecordMeta; // Changed
import borg.trikeshed.nio.IOMemento; // Changed from isam.meta to nio
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Base64;
import java.util.Collections;
import java.util.List;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Utility class for CSV processing, including parsing and type deduction.
 * This class is designed to integrate with the DSEL's {@link Cursor} and {@link TypeMemento} system.
 */
public enum CSVUtil {
    ; // No instances

    /**
     * Parses a single CSV line into a list of strings, handling delimiters and quotes.
     *
     * @param line The CSV line string.
     * @param delimiter The character used to separate fields.
     * @param quote The character used to quote fields.
     * @return A list of strings representing the parsed fields.
     */
    public static @NotNull List<String> parseCsvLine(@Nullable String line, char delimiter, char quote) {
        if (line == null || line.isEmpty()) {
            return Collections.emptyList();
        }

        List<String> fields = new ArrayList<>();
        StringBuilder currentField = new StringBuilder();
        boolean inQuote = false;
        boolean isEscapedQuote = false;

        for (int i = 0; i < line.length(); i++) {
            char c = line.charAt(i);

            if (isEscapedQuote) {
                currentField.append(c);
                isEscapedQuote = false;
            } else if (c == quote) {
                if (inQuote && i + 1 < line.length() && line.charAt(i + 1) == quote) {
                    // Escaped quote (e.g., "" inside a quoted field)
                    currentField.append(quote);
                    i++; // Skip the next quote
                } else {
                    inQuote = !inQuote;
                }
            } else if (c == delimiter && !inQuote) {
                fields.add(currentField.toString());
                currentField = new StringBuilder();
            } else {
                currentField.append(c);
            }
        }
        fields.add(currentField.toString()); // Add the last field

        return fields;
    }

    /**
     * Deduce the most specific type for a given string value.
     * This is a simplified deduction logic.
     *
     * @param value The string value to deduce type for.
     * @return A TypeMemento representing the deduced type.
     */
    public static @NotNull IOMemento deduceType(@Nullable String value) { // Changed return type
        if (value == null || value.trim().isEmpty()) {
            return IOMemento.IoString; // Default for empty/null (IOMemento is now enum)
        }
        String trimmed = value.trim();

        // Try Boolean
        if (trimmed.equalsIgnoreCase("true") || trimmed.equalsIgnoreCase("false")) {
            return IOMemento.IoBoolean;
        }

        // Try numeric types (Long, Double)
        try {
            Long.parseLong(trimmed);
            return IOMemento.IoLong;
        } catch (NumberFormatException e1) {
            try {
                Double.parseDouble(trimmed);
                return IOMemento.IoDouble;
            } catch (NumberFormatException e2) {
                // Not a number
            }
        }

        // Try Char
        if (trimmed.length() == 1) {
            return IOMemento.IoChar;
        }

        // Default to String
        return IOMemento.IoString;
    }

    /**
     * Converts a CSV string value to a typed Java object based on the provided TypeMemento.
     *
     * @param csvString The string value from CSV.
     * @param targetType The TypeMemento to convert to.
     * @return The converted object, or null if conversion fails for numeric/boolean types.
     * @throws IllegalArgumentException if conversion is not possible (e.g., invalid Base64).
     */
    public static @Nullable Object convertCsvStringToTypedValue(@Nullable String csvString, @NotNull IOMemento targetType) { // Changed targetType
        String value = (csvString == null) ? "" : csvString.trim();

        if (targetType instanceof D.FixedSizeTypeMemento) {
            // D.FixedSizeTypeMemento implements IOMemento, getBaseType returns the enum IOMemento
            targetType = ((D.FixedSizeTypeMemento) targetType).getBaseType();
        }
        // targetType is now guaranteed to be the IOMemento enum if it was FixedSizeTypeMemento

        if (value.isEmpty()) {
            // Handle empty strings for different types
            return switch (targetType) { // targetType is IOMemento enum
                case IoBoolean -> false; // Default for empty boolean
                case IoString -> "";
                case IoByteArray -> new byte[0];
                default -> null; // For numeric types, empty string means null
            };
        }

        return switch (targetType) { // targetType is IOMemento enum
            case IoByte -> Byte.parseByte(value);
            case IoShort -> Short.parseShort(value);
            case IoInt -> Integer.parseInt(value);
            case IoLong -> Long.parseLong(value);
            case IoFloat -> Float.parseFloat(value);
            case IoDouble -> Double.parseDouble(value);
            case IoBoolean -> Boolean.parseBoolean(value);
            case IoChar -> {
                if (value.length() == 1) yield value.charAt(0);
                else throw new IllegalArgumentException("Cannot convert '" + value + "' to Char. Expected single character.");
            }
            case IoString -> value;
            case IoByteArray -> {
                try {
                    yield Base64.getDecoder().decode(value);
                } catch (IllegalArgumentException e) {
                    throw new IllegalArgumentException("Invalid Base64 string for IoByteArray: " + value, e);
                }
            }
            case IoInstant, IoLocalDate -> throw new UnsupportedOperationException("Date/Time parsing not implemented for CSVUtil yet.");
        };
    }

    /**
     * Reads a CSV file and converts it into a {@link Cursor} of typed data.
     * This method performs type deduction based on the first few data rows.
     *
     * @param filePath The path to the CSV file.
     * @param hasHeader True if the first line is a header.
     * @param delimiter The field delimiter.
     * @param quote The quote character.
     * @param sampleRowsForTypeDeduction Number of rows to sample for type deduction.
     * @return A Cursor representing the CSV data.
     * @throws IOException If the file cannot be read.
     */
    public static @NotNull Cursor readCsv(@NotNull String filePath, boolean hasHeader, char delimiter, char quote, int sampleRowsForTypeDeduction) throws IOException {
        Path path = Paths.get(filePath);
        List<String> allLinesList = Files.readAllLines(path);
        Series<String> allLinesSeries = Series.of(allLinesList.size(), allLinesList::get);

        if (allLinesSeries.size() == 0) {
            return D.cur(Series.of(0, i -> { throw new IndexOutOfBoundsException(); })); // Use new D.cur with empty Series
        }

        List<String> headerNamesList = hasHeader ? parseCsvLine(allLinesSeries.get(0), delimiter, quote) : Collections.emptyList();
        Series<String> dataLinesSeries = hasHeader ? allLinesSeries.tail(1) : allLinesSeries;

        if (dataLinesSeries.size() == 0) {
            if (hasHeader) {
                // Schema from header, but no data.
                // This part can be complex if we need to create a Cursor with specific schema but no rows.
                // For now, D.cur with empty Series is simplest. If schema must be preserved, this needs more work.
                // List<RecordMeta> headerMeta = headerNamesList.stream().map(name -> D.cm(name, IOMemento.IoString)).collect(Collectors.toList());
                // Series<RecordMeta> schemaSeries = Series.of(headerMeta.size(), headerMeta::get);
                // return D.cur(Series.of(0, i -> { throw new IndexOutOfBoundsException(); }) /*, schemaSeries */); // D.cur doesn't take schema
            }
            return D.cur(Series.of(0, i -> { throw new IndexOutOfBoundsException(); }));
        }

        // 1. Deduce types
        Series<List<String>> parsedSampleRowsSeries = dataLinesSeries.head(sampleRowsForTypeDeduction)
                                                       .alpha(line -> parseCsvLine(line, delimiter, quote));

        int numColumns = parsedSampleRowsSeries.size() > 0 ? parsedSampleRowsSeries.get(0).size() : 0;
        if (numColumns == 0 && parsedSampleRowsSeries.size() > 0) { // if first sample row was empty
             numColumns = parsedSampleRowsSeries.alpha(List::size).toList().stream().mapToInt(Integer::intValue).max().orElse(0);
        }
        if (numColumns == 0) { // Still no columns found
            numColumns = hasHeader ? headerNamesList.size() : 0;
        }

        final int finalNumColumns = numColumns; // For use in lambdas

        IOMemento[] deducedTypesArray = new IOMemento[finalNumColumns];
        for (int colIndex = 0; colIndex < finalNumColumns; colIndex++) {
            final int cIdx = colIndex;
            Series<String> columnValuesSeries = parsedSampleRowsSeries.alpha(
                parsedRowList -> cIdx < parsedRowList.size() ? parsedRowList.get(cIdx) : ""
            );

            IOMemento commonType = IOMemento.IoString;
            int maxStrLength = 0;
            for (int i = 0; i < columnValuesSeries.size(); i++) {
                String val = columnValuesSeries.get(i);
                IOMemento currentValType = deduceType(val);
                if (currentValType == IOMemento.IoString) {
                    commonType = IOMemento.IoString;
                } else if (currentValType == IOMemento.IoDouble && commonType != IOMemento.IoString) {
                    commonType = IOMemento.IoDouble;
                } else if (currentValType == IOMemento.IoLong && commonType == IOMemento.IoBoolean) {
                    commonType = IOMemento.IoLong;
                } else if (commonType == IOMemento.IoBoolean && currentValType != IOMemento.IoBoolean) {
                     commonType = currentValType;
                }
                maxStrLength = Math.max(maxStrLength, val != null ? val.length() : 0);
            }

            if (commonType == IOMemento.IoString) {
                deducedTypesArray[cIdx] = D.fsString(Math.max(1, maxStrLength + 1));
            } else {
                deducedTypesArray[cIdx] = commonType;
            }
        }
        Series<IOMemento> deducedTypesSeries = Series.of(deducedTypesArray.length, idx -> deducedTypesArray[idx]);

        // 2. Create RecordMeta for the schema
        RecordMeta[] schemaArray = new RecordMeta[finalNumColumns];
        for (int colIndex = 0; colIndex < finalNumColumns; colIndex++) {
            String colName = (hasHeader && colIndex < headerNamesList.size()) ? headerNamesList.get(colIndex) : "column_" + colIndex;
            IOMemento colType = deducedTypesSeries.get(colIndex);
            schemaArray[colIndex] = D.cm(colName, colType);
        }
        Series<RecordMeta> schemaSeries = Series.of(schemaArray.length, idx -> schemaArray[idx]);

        // 3. Process all data rows into RowVecs
        Series<RowVec> rowsSeries = dataLinesSeries.alpha(line -> {
            List<String> parsedFields = parseCsvLine(line, delimiter, quote);
            @SuppressWarnings("unchecked") // For Join[] array creation
            Join<Object, Supplier<RecordMeta>>[] rowCellsArray = new Join[finalNumColumns];
            for (int colIndex = 0; colIndex < finalNumColumns; colIndex++) {
                String stringValue = colIndex < parsedFields.size() ? parsedFields.get(colIndex) : "";
                IOMemento colType = schemaSeries.get(colIndex).type();
                Object typedValue = convertCsvStringToTypedValue(stringValue, colType);
                final int cIdx = colIndex;
                rowCellsArray[colIndex] = D.jn(typedValue, (Supplier<RecordMeta>) () -> schemaSeries.get(cIdx));
            }
            return D.rv(rowCellsArray);
        });

        return D.cur(rowsSeries); // Use new D.cur that takes Series<RowVec>
    }
}
