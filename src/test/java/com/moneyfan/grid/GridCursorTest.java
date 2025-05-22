package com.moneyfan.grid;

import com.moneyfan.core.CellMeta;
import com.moneyfan.core.IOMemento;
import com.moneyfan.core.Scalar;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.function.Supplier;

import static org.junit.jupiter.api.Assertions.*;

class GridCursorTest {

    private static Cell cell(Object value, Scalar scalar) {
        Supplier<Scalar> sup = () -> scalar;
        return new Cell(value, new CellMeta(sup));
    }

    @Test
    void basicAccessorsWork() {
        Scalar idScalar = new Scalar(IOMemento.IO_INT, "id");
        Scalar nameScalar = new Scalar(IOMemento.IO_STRING_FIXED, "name");

        RowVec row = new RowVec(
                Vect0r.fromList(List.of(
                        cell(1, idScalar),
                        cell("Alice", nameScalar)
                )));

        GridCursor grid = new GridCursor(Vect0r.fromList(List.of(row)));

        assertEquals(1, grid.rowCount());
        assertEquals(2, grid.columnCount());
        assertEquals("Alice", grid.getRow(0).cells().get(1).value());

        Vect0r<Scalar> scalars = grid.getScalars();
        assertEquals("id", scalars.get(0).name());
        assertEquals("name", scalars.get(1).name());
    }
}