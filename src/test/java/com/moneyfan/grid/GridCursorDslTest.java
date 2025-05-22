package com.moneyfan.grid;

import com.moneyfan.core.IOMemento;
import com.moneyfan.core.Scalar;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

public class GridCursorDslTest {

    private static GridCursor buildGrid() {
        List<Scalar> schema = List.of(
                Scalar.of(IOMemento.IO_INT, "id"),
                Scalar.of(IOMemento.IO_STRING_FIXED, "name"),
                Scalar.of(IOMemento.IO_DOUBLE, "value")
        );
        RowVec r1 = makeRow(1, "alice", 10.0, schema);
        RowVec r2 = makeRow(2, "bob", 20.0, schema);
        RowVec r3 = makeRow(3, "carol", 30.0, schema);
        return new GridCursor(Vect0r.fromList(List.of(r1, r2, r3)));
    }

    private static RowVec makeRow(int id, String name, double value, List<Scalar> schema) {
        List<Cell> cells = List.of(
                new Cell(id, new com.moneyfan.core.CellMeta(() -> schema.get(0))),
                new Cell(name, new com.moneyfan.core.CellMeta(() -> schema.get(1))),
                new Cell(value, new com.moneyfan.core.CellMeta(() -> schema.get(2)))
        );
        return new RowVec(Vect0r.fromList(cells));
    }

    @Test
    void selectColumns() {
        GridCursor grid = buildGrid();
        GridCursor selected = grid.select("name", "value");
        assertEquals(2, selected.columnCount());
        assertEquals("bob", selected.getRow(1).get(0).value());
    }

    @Test
    void mapColumn() {
        GridCursor grid = buildGrid();
        GridCursor mapped = grid.mapColumn("value", IOMemento.IO_DOUBLE, v -> ((Double) v) * 2);
        assertEquals(60.0, (Double) mapped.getRow(2).get(2).value(), 1e-6);
    }

    @Test
    void filterRows() {
        GridCursor grid = buildGrid();
        GridCursor filtered = grid.filter(row -> (Integer) row.get(0).value() > 1);
        assertEquals(2, filtered.rowCount());
        assertEquals("bob", filtered.getRow(0).get(1).value());
    }
}