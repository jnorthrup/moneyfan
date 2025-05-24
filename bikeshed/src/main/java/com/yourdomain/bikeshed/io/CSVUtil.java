package com.yourdomain.bikeshed.io;

import com.yourdomain.bikeshed.core.Cursor;
import com.yourdomain.bikeshed.core.Join;
import com.yourdomain.bikeshed.core.RowVec;
import com.yourdomain.bikeshed.core.Series;
import com.yourdomain.bikeshed.dsel.D;
import com.yourdomain.bikeshed.type.ColumnMeta;
import com.yourdomain.bikeshed.type.TypeMemento;
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
    public static @NotNull TypeMemento deduceType(@Nullable String value) {
        if (value == null || value.trim().isEmpty()) {
            return IOMemento.IoString; // Default for empty/null
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
    public static @Nullable Object convertCsvStringToTypedValue(@Nullable String csvString, @NotNull TypeMemento targetType) {
        String value = (csvString == null) ? "" : csvString.trim();

        if (targetType instanceof D.FixedSizeTypeMemento) {
            targetType = ((D.FixedSizeTypeMemento) targetType).getBaseType();
        }

        if (value.isEmpty()) {
            // Handle empty strings for different types
            return switch ((IOMemento) targetType) {
                case IoBoolean -> false; // Default for empty boolean
                case IoString -> "";
                case IoByteArray -> new byte[0];
                default -> null; // For numeric types, empty string means null
            };
        }

        return switch ((IOMemento) targetType) {
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
        List<String> allLines = Files.readAllLines(path);

        if (allLines.isEmpty()) {
            return Cursor.of(Collections.emptyList());
        }

        List<String> headerLine = hasHeader ? parseCsvLine(allLines.get(0), delimiter, quote) : Collections.emptyList();
        List<String> dataLines = hasHeader ? allLines.subList(1, allLines.size()) : allLines;

        if (dataLines.isEmpty()) {
            // If no data lines, but had header, create empty cursor with header schema
            if (hasHeader) {
                List<ColumnMeta> headerMeta = headerLine.stream()
                        .map(name -> D.cm(name, IOMemento.IoString)) // Default to String for empty data
                        .collect(Collectors.toList());
                return Cursor.of(Collections.emptyList()).selectColumns(IntStream.range(0, headerMeta.size()).toArray()); // Placeholder for schema
            }
            return Cursor.of(Collections.emptyList());
        }

        // 1. Deduce types
        List<List<String>> parsedSampleRows = dataLines.stream()
                .limit(sampleRowsForTypeDeduction)
                .map(line -> parseCsvLine(line, delimiter, quote))
                .collect(Collectors.toList());

        int numColumns = parsedSampleRows.stream().mapToInt(List::size).max().orElse(0);
        if (numColumns == 0) { // No data rows or empty rows
            numColumns = hasHeader ? headerLine.size() : 0;
        }

        List<TypeMemento> deducedTypes = IntStream.range(0, numColumns)
                .mapToObj(colIndex -> {
                    // Collect all values for this column from sample rows
                    List<String> columnValues = parsedSampleRows.stream()
                            .filter(row -> colIndex < row.size())
                            .map(row -> row.get(colIndex))
                            .collect(Collectors.toList());

                    // Deduce the most general type that fits all samples in this column
                    TypeMemento commonType = IOMemento.IoString; // Start with most general
                    int maxStrLength = 0;

                    for (String val : columnValues) {
                        TypeMemento currentValType = deduceType(val);
                        if (currentValType == IOMemento.IoString) {
                            commonType = IOMemento.IoString; // If any is string, it's a string column
                        } else if (currentValType == IOMemento.IoDouble && commonType != IOMemento.IoString) {
                            commonType = IOMemento.IoDouble; // If any is double, and not string, it's double
                        } else if (currentValType == IOMemento.IoLong && commonType == IOMemento.IoBoolean) {
                            commonType = IOMemento.IoLong; // Long overrides boolean
                        }
                        maxStrLength = Math.max(maxStrLength, val != null ? val.length() : 0);
                    }

                    // For string types, use FixedSizeTypeMemento with max length + 1 for null terminator/padding
                    if (commonType == IOMemento.IoString) {
                        return D.fsString(Math.max(1, maxStrLength + 1));
                    }
                    return commonType;
                })
                .collect(Collectors.toList());

        // 2. Create ColumnMeta for the schema
        List<ColumnMeta> schema = IntStream.range(0, numColumns)
                .mapToObj(colIndex -> {
                    String colName = hasHeader && colIndex < headerLine.size() ? headerLine.get(colIndex) : "column_" + colIndex;
                    TypeMemento colType = deducedTypes.get(colIndex);
                    return D.cm(colName, colType);
                })
                .collect(Collectors.toList());

        // 3. Process all data rows into RowVecs
        List<RowVec> rows = dataLines.stream()
                .map(line -> {
                    List<String> parsedFields = parseCsvLine(line, delimiter, quote);
                    List<Join<Object, Supplier<ColumnMeta>>> rowCells = IntStream.range(0, numColumns)
                            .mapToObj(colIndex -> {
                                String stringValue = colIndex < parsedFields.size() ? parsedFields.get(colIndex) : "";
                                TypeMemento colType = schema.get(colIndex).type();
                                Object typedValue = convertCsvStringToTypedValue(stringValue, colType);
                                return D.jn(typedValue, (Supplier<ColumnMeta>) () -> schema.get(colIndex));
                            })
                            .collect(Collectors.toList());
                    return D.rv(rowCells);
                })
                .collect(Collectors.toList());

        return D.cur(rows);
    }
}
