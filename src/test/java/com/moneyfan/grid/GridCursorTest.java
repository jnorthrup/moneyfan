package com.moneyfan.grid;

import com.moneyfan.core.CellMeta;
import com.moneyfan.core.IOMemento;
import com.moneyfan.core.Scalar;
import org.junit.jupiter.api.Test;

import java.util.function.Supplier;

import static org.junit.jupiter.api.Assertions.*;

class GridCursorTest {

    private static CellMeta meta(Scalar s) {
        Supplier<Scalar> sup = () -> s;
        return new CellMeta(sup);
    }

    @Test
    void basicConstructionAndAccess() {
        Scalar scInt = Scalar.of(IOMemento.INT, "id");
        Scalar scStr = Scalar.fixedString("name", 8);

        RowVec row0 = new RowVec(Vect0r.of(
                new Cell(1, meta(scInt)),
                new Cell("Alice", meta(scStr))
        ));

        RowVec row1 = new RowVec(Vect0r.of(
                new Cell(2, meta(scInt)),
                new Cell("Bob", meta(scStr))
        ));

        GridCursor grid = new GridCursor(Vect0r.of(row0, row1));

        assertEquals(2, grid.rowCount());
        assertEquals(2, grid.columnCount());

        assertEquals("Bob", grid.row(1).value(1));
        assertEquals(IOMemento.INT, grid.scalars().get(0).type());
    }
}