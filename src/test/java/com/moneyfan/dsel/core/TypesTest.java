package com.moneyfan.dsel.core;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import java.util.function.Function;
import java.util.function.Supplier;

import static com.moneyfan.dsel.core.Types.*; // Import static factory methods

class TypesTest {

    @Test
    void testJoinCreation() {
        Join<String, Integer> j1 = jn("test", 1);
        assertEquals("test", j1.f());
        assertEquals(1, j1.s());

        Join<String, Integer> j2 = Join.jn("test", 1); // Direct from Join record
        assertEquals(j1, j2);
    }

    @Test
    void testColumnMetaCreation() {
        Join<String, TypeMemento> cm1 = cm("age", TypeMemento.Basic.INTEGER);
        assertEquals("age", cm1.f());
        assertEquals(TypeMemento.Basic.INTEGER, cm1.s());
    }

    @Test
    void testSeriesCreationAndAccess() {
        Join<Integer, Function<Integer, String>> mySeries = sr(3, i -> "val" + i);

        assertEquals(3, size(mySeries));
        assertEquals("val0", get(mySeries, 0));
        assertEquals("val1", get(mySeries, 1));
        assertEquals("val2", get(mySeries, 2));

        assertThrows(IndexOutOfBoundsException.class, () -> get(mySeries, -1));
        assertThrows(IndexOutOfBoundsException.class, () -> get(mySeries, 3));
        assertThrows(IllegalArgumentException.class, () -> sr(-1, i -> ""));
    }

    @Test
    void testRowVecConcept() {
        // Define a cell: Join<Object, Supplier<ColumnMeta>>
        // ColumnMeta: Join<String, TypeMemento>
        Supplier<Join<String, TypeMemento>> nameColMetaSupplier = () -> cm("name", TypeMemento.Basic.STRING);
        Join<Object, Supplier<Join<String, TypeMemento>>> cell1 = jn("Alice", nameColMetaSupplier);

        Supplier<Join<String, TypeMemento>> ageColMetaSupplier = () -> cm("age", TypeMemento.Basic.INTEGER);
        Join<Object, Supplier<Join<String, TypeMemento>>> cell2 = jn(30, ageColMetaSupplier);

        // RowVec: Series<CellType>
        // RowVec actual type: Join<Integer, Function<Integer, Join<Object, Supplier<Join<String, TypeMemento>>>>>
        Join<Integer, Function<Integer, Join<Object, Supplier<Join<String, TypeMemento>>>>> rowVector =
                rv(2, idx -> idx == 0 ? cell1 : cell2);

        assertEquals(2, size(rowVector));
        assertEquals("Alice", get(rowVector, 0).f());
        assertEquals("name", get(rowVector, 0).s().get().f());
        assertEquals(30, get(rowVector, 1).f());
        assertEquals("age", get(rowVector, 1).s().get().f());
    }

    @Test
    void testTwinCreation() {
        Join<String, String> nameTwin = tw("FirstName", "LastName");
        assertEquals("FirstName", nameTwin.f());
        assertEquals("LastName", nameTwin.s());
    }
}
