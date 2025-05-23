package com.moneyfan;

import com.moneyfan.dsel.D;
import com.moneyfan.dsel.core.ColumnMeta;
import com.moneyfan.dsel.core.Cursor;
import com.moneyfan.dsel.core.RowVec;
import com.moneyfan.dsel.core.TypeMemento;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;

public class App {
    public static void main(String[] args) {
        System.out.println("DSEL Demo: ISAM and CSV");

        // Define a schema
        List<ColumnMeta> schema = Arrays.asList(
            D.cm("ID", TypeMemento.Basic.INTEGER),
            D.cm("Name", TypeMemento.Basic.STRING), // For fixed-size ISAM, STRING needs a fixed size
            D.cm("Value", TypeMemento.Basic.DOUBLE)
        );
        // Adjust STRING for fixed-size ISAM
        List<ColumnMeta> fixedSchema = Arrays.asList(
            D.cm("ID", TypeMemento.Basic.INTEGER),
            D.cm("Name", new TypeMemento() { // Custom fixed-size string
                @Override public String getTypeName() { return "FixedString10"; }
                @Override public int getFixedSize() { return 10; } // String of 10 bytes
            }),
            D.cm("Value", TypeMemento.Basic.DOUBLE)
        );


        // Create some data using DSEL
        Cursor dataCursor = D.cr(2, rowIndex ->
            D.rv(3, colIndex -> {
                Object val = switch (colIndex) {
                    case 0 -> rowIndex + 1;
                    case 1 -> "Item" + rowIndex;
                    case 2 -> (rowIndex + 1) * 10.5;
                    default -> null;
                };
                // Use fixedSchema for ISAM compatibility
                return D.jn(val, () -> fixedSchema.get(colIndex));
            })
        );

        System.out.println("Original Data:");
        D.each(dataCursor, row -> {
            D.each(row, cell -> System.out.print(D.f(cell) + " ("+ D.f(D.s(cell).get()) +") | "));
            System.out.println();
        });


        // --- ISAM Example ---
        String isamPathBase = "test_data";
        try {
            System.out.println("\nWriting to ISAM: " + isamPathBase);
            D.writeIsam(isamPathBase, dataCursor);

            System.out.println("Reading from ISAM:");
            D.IsamCursor isamCursor = null;
            try {
                isamCursor = new D.IsamCursor(isamPathBase);
                D.each((Cursor)isamCursor, row -> {
                    System.out.print("ISAM Row: ");
                    D.each((RowVec)row, cell -> System.out.print(D.f((com.moneyfan.dsel.core.Join<?, ?>)cell) + " | "));
                    System.out.println();
                });
            } finally {
                if (isamCursor != null) {
                    try {
                        isamCursor.close();
                    } catch (IOException e) {
                        System.err.println("Error closing ISAM cursor: " + e.getMessage());
                    }
                }
            }
        } catch (IOException e) {
            System.err.println("ISAM Error: " + e.getMessage());
            e.printStackTrace();
        } finally {
            try { Files.deleteIfExists(Path.of(isamPathBase + D.DATA_SFX)); } catch (IOException e) {}
            try { Files.deleteIfExists(Path.of(isamPathBase + D.META_SFX)); } catch (IOException e) {}
        }

        // --- CSV Example ---
        String csvFilePath = "test_data.csv";
        try {
            // Create a dummy CSV file
            List<String> csvLines = Arrays.asList(
                "ID,Name,Value", // Header
                "10,ProductA,100.50",
                "20,ProductB,200.75"
            );
            Files.write(Path.of(csvFilePath), csvLines);

            System.out.println("\nReading from CSV: " + csvFilePath);
            // Use schema for parsing (could be inferred too, but explicit is better for type safety)
            // Note: CSV parsing uses the original 'schema' which might have variable-length strings.
            // For writing this to ISAM, it would need transformation or a schema with fixed sizes.
            Cursor csvCursor = D.readCsv(csvFilePath, schema, ",", true);
            D.each(csvCursor, row -> {
                System.out.print("CSV Row: ");
                D.each(row, cell -> System.out.print(D.f(cell) + " | "));
                System.out.println();
            });

            // Example: Convert CSV to ISAM (using fixedSchema for the ISAM output)
            // This requires that the data from CSV can be meaningfully converted to fixedSchema.
            // For simplicity, we'll re-create a cursor that fits fixedSchema from CSV-like strings.
            List<ColumnMeta> csvReadSchema = Arrays.asList( // Schema matching the CSV for reading
                D.cm("ID", TypeMemento.Basic.INTEGER),
                D.cm("Name", TypeMemento.Basic.STRING), // Read as variable string
                D.cm("Value", TypeMemento.Basic.DOUBLE)
            );
            Cursor rawCsvCursor = D.readCsv(csvFilePath, csvReadSchema, ",", true);

            // Transform rawCsvCursor (with variable strings) to a cursor compatible with fixedSchema
            Cursor forIsamCursor = D.mapRow(rawCsvCursor, rawRow ->
                D.rv(D.sz(rawRow), colIdx -> {
                    Object rawVal = D.get(rawRow, colIdx);
                    ColumnMeta targetMeta = fixedSchema.get(colIdx); // Target ISAM schema
                    Object convertedVal = rawVal;
                    if (D.colName(rawRow, colIdx).equals("Name") && rawVal instanceof String) {
                        // Truncate/pad string for fixed-size field
                        String s = (String) rawVal;
                        int fixedLen = targetMeta.s().getFixedSize();
                        convertedVal = s.length() > fixedLen ? s.substring(0, fixedLen) : String.format("%-" + fixedLen + "s", s).replace(' ', '\0'); // Pad with nulls
                    }
                    return D.jn(convertedVal, () -> targetMeta);
                })
            );

            String csvToIsamPathBase = "csv_to_isam_data";
            System.out.println("\nConverting CSV to ISAM: " + csvToIsamPathBase);
            D.csvToIsam(csvFilePath, csvToIsamPathBase, fixedSchema, ",", true); // Using fixedSchema definition for output ISAM

            System.out.println("Reading from CSV-converted ISAM:");
            D.IsamCursor convertedIsamCursor = null;
            try {
                convertedIsamCursor = new D.IsamCursor(csvToIsamPathBase);
                D.each((Cursor)convertedIsamCursor, row -> {
                    System.out.print("ISAM(CSV) Row: ");
                    D.each((RowVec)row, cell -> System.out.print( (""+D.f((com.moneyfan.dsel.core.Join<?, ?>)cell)).replace('\0', ' ') + " | ")); // Replace null padding for print
                    System.out.println();
                });
            } finally {
                if (convertedIsamCursor != null) {
                    try {
                        convertedIsamCursor.close();
                    } catch (IOException e) {
                        System.err.println("Error closing ISAM cursor: " + e.getMessage());
                    }
                }
            }


        } catch (IOException e) {
            System.err.println("CSV/ISAM Error: " + e.getMessage());
            e.printStackTrace();
        } finally {
             try { Files.deleteIfExists(Path.of(csvFilePath)); } catch (IOException e) {}
             try { Files.deleteIfExists(Path.of("csv_to_isam_data" + D.DATA_SFX)); } catch (IOException e) {}
             try { Files.deleteIfExists(Path.of("csv_to_isam_data" + D.META_SFX)); } catch (IOException e) {}
        }
    }
}