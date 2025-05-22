package com.moneyfan;

import com.moneyfan.core.CellMeta;
import com.moneyfan.core.IOMemento;
import com.moneyfan.core.Scalar;
import com.moneyfan.grid.Cell;
import com.moneyfan.grid.GridCursor;
import com.moneyfan.grid.RowVec;
import com.moneyfan.grid.Vect0r;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

public class GridCursorTest {
    
    @Test
    public void testBasicGridCursorOperations() {
        // Define test schema
        Scalar idScalar = Scalar.of(IOMemento.IO_INT, "id");
        Scalar nameScalar = Scalar.stringOf("name", 10);
        Scalar valueScalar = Scalar.of(IOMemento.IO_DOUBLE, "value");
        
        // Create test data
        List<RowVec> rows = new ArrayList<>();
        
        // Row 1
        List<Cell> row1Cells = List.of(
            Cell.of(1, CellMeta.of(idScalar)),
            Cell.of("Apple", CellMeta.of(nameScalar)),
            Cell.of(10.5, CellMeta.of(valueScalar))
        );
        rows.add(RowVec.of(Vect0r.fromList(row1Cells)));
        
        // Row 2
        List<Cell> row2Cells = List.of(
            Cell.of(2, CellMeta.of(idScalar)),
            Cell.of("Banana", CellMeta.of(nameScalar)),
            Cell.of(5.2, CellMeta.of(valueScalar))
        );
        rows.add(RowVec.of(Vect0r.fromList(row2Cells)));
        
        // Row 3
        List<Cell> row3Cells = List.of(
            Cell.of(3, CellMeta.of(idScalar)),
            Cell.of("Cherry", CellMeta.of(nameScalar)),
            Cell.of(8.7, CellMeta.of(valueScalar))
        );
        rows.add(RowVec.of(Vect0r.fromList(row3Cells)));
        
        // Create GridCursor
        GridCursor cursor = GridCursor.of(Vect0r.fromList(rows));
        
        // Test basic properties
        assertEquals(3, cursor.rowCount());
        assertEquals(3, cursor.columnCount());
        assertEquals(3, cursor.getScalars().size());
        
        // Test row access
        RowVec row1 = cursor.getRow(0);
        assertEquals(1, row1.getValue(0));
        assertEquals("Apple", row1.getValue(1));
        assertEquals(10.5, row1.getValue(2));
        
        // Test select operation
        GridCursor selectedCursor = cursor.select("id", "value");
        assertEquals(3, selectedCursor.rowCount());
        assertEquals(2, selectedCursor.columnCount());
        assertEquals(1, selectedCursor.getRow(0).getValue(0));
        assertEquals(10.5, selectedCursor.getRow(0).getValue(1));
        
        // Test filter operation
        GridCursor filteredCursor = cursor.filter(row -> 
            ((Double) row.getValue(2)) > 7.0
        );
        assertEquals(2, filteredCursor.rowCount());
        assertEquals("Apple", filteredCursor.getRow(0).getValue(1));
        assertEquals("Cherry", filteredCursor.getRow(1).getValue(1));
    }
}