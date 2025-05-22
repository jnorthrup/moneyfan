package com.moneyfan.io;

import com.moneyfan.core.CellMeta;
import com.moneyfan.core.Scalar;
import com.moneyfan.grid.Cell;
import com.moneyfan.grid.GridCursor;
import com.moneyfan.grid.RowVec;
import com.moneyfan.grid.Vect0r;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Utility to read CSV files into an in-memory GridCursor.
 */
public class CSVCursorReader {
    
    private final Path csvPath;
    private final List<Scalar> schema;
    private final boolean hasHeader;
    
    public CSVCursorReader(Path csvPath, List<Scalar> schema, boolean hasHeader) {
        this.csvPath = csvPath;
        this.schema = schema;
        this.hasHeader = hasHeader;
    }
    
    public GridCursor read() throws IOException {
        List<RowVec> rows = new ArrayList<>();
        
        try (BufferedReader reader = Files.newBufferedReader(csvPath)) {
            // Skip header if needed
            if (hasHeader) {
                reader.readLine();
            }
            
            String line;
            while ((line = reader.readLine()) != null) {
                String[] values = line.split(",", -1);
                if (values.length < schema.size()) {
                    continue; // Skip malformed rows
                }
                
                List<Cell> cells = new ArrayList<>(schema.size());
                
                for (int i = 0; i < schema.size(); i++) {
                    Scalar scalar = schema.get(i);
                    String value = values[i].trim();
                    
                    Object typedValue;
                    if (value.isEmpty()) {
                        typedValue = null;
                    } else {
                        typedValue = scalar.type().convertFromString(value);
                    }
                    
                    cells.add(Cell.of(typedValue, CellMeta.of(scalar)));
                }
                
                rows.add(RowVec.of(Vect0r.fromList(cells)));
            }
        }
        
        return GridCursor.of(Vect0r.fromList(rows));
    }
    
    public static GridCursor readFromCSV(Path csvPath, List<Scalar> schema, boolean hasHeader) throws IOException {
        return new CSVCursorReader(csvPath, schema, hasHeader).read();
    }
}