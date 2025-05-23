package com.moneyfan.dsel;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.NoSuchElementException;

import static org.junit.jupiter.api.Assertions.*;

class DSEL_CursorTest {

    private DSEL_Cursor<String, Integer> emptyCursor;
    private DSEL_Cursor<String, Integer> singleElementCursor;
    private DSEL_Cursor<String, Integer> multiElementCursor;

    @BeforeEach
    void setUp() {
        emptyCursor = DSEL.INSTANCE.of();
        singleElementCursor = DSEL.INSTANCE.of(Join.of("A", 1));
        multiElementCursor = DSEL.INSTANCE.of(
            Join.of("Alice", 30),
            Join.of("Bob", 25),
            Join.of("Charlie", 35)
        );
    }

    @Test
    void testMapFirst() {
        DSEL_Cursor<Integer, Integer> lengths = multiElementCursor.mfst(String::length);
        assertEquals(3, lengths.count());
        assertEquals(List.of(Join.of(5, 30), Join.of(3, 25), Join.of(7, 35)), lengths.collect());
        assertTrue(emptyCursor.mfst(String::length).isEmpty());
    }

    @Test
    void testMapSecond() {
        DSEL_Cursor<String, Integer> incrementedAges = multiElementCursor.msnd(age -> age + 1);
        assertEquals(3, incrementedAges.count());
        assertEquals(List.of(Join.of("Alice", 31), Join.of("Bob", 26), Join.of("Charlie", 36)), incrementedAges.collect());
    }

    @Test
    void testMapBoth() {
        DSEL_Cursor<Integer, String> transformed = multiElementCursor.mbth(
            String::length,
            age -> "Age:" + age
        );
        assertEquals(List.of(Join.of(5, "Age:30"), Join.of(3, "Age:25"), Join.of(7, "Age:35")), transformed.collect());
    }

    @Test
    void testSwap() {
        DSEL_Cursor<Integer, String> swapped = multiElementCursor.swp();
        assertEquals(List.of(Join.of(30, "Alice"), Join.of(25, "Bob"), Join.of(35, "Charlie")), swapped.collect());
    }

    @Test
    void testFilter() {
        DSEL_Cursor<String, Integer> filtered = multiElementCursor.flt(join -> join.second() > 25);
        assertEquals(2, filtered.count());
        assertEquals(List.of(Join.of("Alice", 30), Join.of("Charlie", 35)), filtered.collect());
    }

    @Test
    void testFilterFirst() {
        DSEL_Cursor<String, Integer> filtered = multiElementCursor.fltFst(name -> name.startsWith("A"));
        assertEquals(List.of(Join.of("Alice", 30)), filtered.collect());
    }

    @Test
    void testFilterSecond() {
        DSEL_Cursor<String, Integer> filtered = multiElementCursor.fltSnd(age -> age < 30);
        assertEquals(List.of(Join.of("Bob", 25)), filtered.collect());
    }

    @Test
    void testCollect() {
        List<Join<String, Integer>> collected = multiElementCursor.col();
        assertEquals(3, collected.size());
        assertEquals(Join.of("Bob", 25), collected.get(1));
    }

    @Test
    void testCount() {
        assertEquals(0, emptyCursor.cnt());
        assertEquals(1, singleElementCursor.cnt());
        assertEquals(3, multiElementCursor.cnt());
    }

    @Test
    void testIsEmpty() {
        assertTrue(emptyCursor.isEmp());
        assertFalse(singleElementCursor.isEmp());
    }

    @Test
    void testFirstJoin() {
        assertEquals(Join.of("A",1), singleElementCursor.fstJ());
        assertThrows(NoSuchElementException.class, () -> emptyCursor.firstJoin());
    }
}
