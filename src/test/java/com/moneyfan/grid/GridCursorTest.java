package com.moneyfan.grid;

import com.moneyfan.core.*;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.function.Supplier;

import static org.junit.jupiter.api.Assertions.*;

public class GridCursorTest {

    private static Cell makeCell(Object value, Scalar scalar) {
        Supplier<Scalar> supplier = () -> scalar;
        return new Cell(value, new CellMeta(supplier));
    }

    @Test
    void basicGridCursorFunctions() {
        Scalar idScalar = Scalar.of(IOMemento.IO_INT, "id");
        Scalar nameScalar = Scalar.of(IOMemento.IO_STRING_FIXED, "name");

        // build two rows
        RowVec row1 = new RowVec(Vect0r.fromList(List.of(
                makeCell(1, idScalar),
                makeCell("alice", nameScalar)
        )));
        RowVec row2 = new RowVec(Vect0r.fromList(List.of(
                makeCell(2, idScalar),
                makeCell("bob", nameScalar)
        )));

        GridCursor grid = new GridCursor(Vect0r.fromList(List.of(row1, row2)));

        assertEquals(2, grid.rowCount());
        assertEquals(2, grid.columnCount());
        assertEquals(1, grid.getRow(0).get(0).value());
        assertEquals("bob", grid.getRow(1).get(1).value());

        List<Scalar> scalars = grid.getScalars();
        assertEquals(2, scalars.size());
        assertEquals("id", scalars.get(0).name());
        assertEquals(IOMemento.IO_STRING_FIXED, scalars.get(1).type());
    }
}