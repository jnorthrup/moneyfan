package com.moneyfan.grid;

import com.moneyfan.core.IOMemento;
import com.moneyfan.core.Scalar;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

public class JoinTest {

    private static RowVec makeRow(List<Object> values, List<Scalar> schema) {
        java.util.List<Cell> cells = new java.util.ArrayList<>();
        for(int i=0;i<values.size();i++) {
            Object v = values.get(i);
            Scalar sc = schema.get(i);
            cells.add(new Cell(v, new com.moneyfan.core.CellMeta(() -> sc)));
        }
        return new RowVec(Vect0r.fromList(cells));
    }

    @Test
    void innerJoinOnId() {
        List<Scalar> schemaA = List.of(
                Scalar.of(IOMemento.IO_INT, "id"),
                Scalar.of(IOMemento.IO_STRING_FIXED, "name")
        );
        List<Scalar> schemaB = List.of(
                Scalar.of(IOMemento.IO_INT, "id"),
                Scalar.of(IOMemento.IO_DOUBLE, "value")
        );
        GridCursor left = new GridCursor(Vect0r.fromList(List.of(
                makeRow(List.of(1, "alice"), schemaA),
                makeRow(List.of(2, "bob"), schemaA)
        )));
        GridCursor right = new GridCursor(Vect0r.fromList(List.of(
                makeRow(List.of(2, 20.0), schemaB),
                makeRow(List.of(3, 30.0), schemaB)
        )));
        GridCursor joined = left.innerJoin(right, "id");
        assertEquals(1, joined.rowCount());
        assertEquals("bob", joined.getRow(0).get(1).value());
        assertEquals(20.0, (Double) joined.getRow(0).get(3).value(), 1e-6);
    }
}