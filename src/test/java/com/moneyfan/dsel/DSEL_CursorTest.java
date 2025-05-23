package com.moneyfan.dsel;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import static com.moneyfan.dsel.JoinOps.cj;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Optional;

class DSEL_CursorTest {

    @Test
    void testMapFirst() {
        DSEL_Cursor<String, Integer> multiElementCursor = new ListBackedCursor<>(Arrays.asList(
                cj("Alice", 25),
                cj("Bob", 30),
                cj("Charlie", 35)
        ));

        DSEL_Cursor<Integer, Integer> mappedCursor = multiElementCursor
                .mf(name -> name.length());

        List<Join<Integer, Integer>> expected = Arrays.asList(
                cj(5, 25),
                cj(3, 30),
                cj(7, 35)
        );
        assertEquals(expected, mappedCursor.collect());

        DSEL_Cursor<String, Integer> emptyCursor = new ListBackedCursor<>(Collections.emptyList());
        DSEL_Cursor<Integer, Integer> mappedEmptyCursor = emptyCursor
                .mf(String::length);
        assertTrue(mappedEmptyCursor.collect().isEmpty());

        DSEL_Cursor<String, Integer> singleElementCursor = new ListBackedCursor<>(Collections.singletonList(
                cj("Alice", 25)
        ));
        DSEL_Cursor<Integer, Integer> mappedSingleCursor = singleElementCursor
                .mf(String::length);
        assertEquals(Collections.singletonList(
                cj(5, 25)
        ), mappedSingleCursor.collect());
    }

    @Test
    void testMapSecond() {
        DSEL_Cursor<String, Integer> multiElementCursor = new ListBackedCursor<>(Arrays.asList(
                cj("Alice", 25),
                cj("Bob", 30),
                cj("Charlie", 35)
        ));

        DSEL_Cursor<String, Integer> mappedCursor = multiElementCursor
                .ms(age -> age + 1);

        List<Join<String, Integer>> expected = Arrays.asList(
                cj("Alice", 26),
                cj("Bob", 31),
                cj("Charlie", 36)
        );
        assertEquals(expected, mappedCursor.collect());
    }

    @Test
    void testMapBoth() {
        DSEL_Cursor<String, Integer> multiElementCursor = new ListBackedCursor<>(Arrays.asList(
                cj("Alice", 25),
                cj("Bob", 30),
                cj("Charlie", 35)
        ));

        DSEL_Cursor<Integer, String> mappedCursor = multiElementCursor
                .mb((name, age) -> cj(name.length(), "Age:" + age));

        List<Join<Integer, String>> expected = Arrays.asList(
                cj(5, "Age:25"),
                cj(3, "Age:30"),
                cj(7, "Age:35")
        );
        assertEquals(expected, mappedCursor.collect());
    }

    @Test
    void testFilter() {
        DSEL_Cursor<String, Integer> multiElementCursor = new ListBackedCursor<>(Arrays.asList(
                cj("Alice", 25),
                cj("Bob", 30),
                cj("Charlie", 35)
        ));

        DSEL_Cursor<String, Integer> filteredCursor = multiElementCursor
                .fl(join -> join.second() > 25);

        List<Join<String, Integer>> expected = Arrays.asList(
                cj("Bob", 30),
                cj("Charlie", 35)
        );
        assertEquals(expected, filteredCursor.collect());

        DSEL_Cursor<String, Integer> filteredByFirst = multiElementCursor
                .fl(join -> join.first().startsWith("A"));
        assertEquals(Collections.singletonList(
                cj("Alice", 25)
        ), filteredByFirst.collect());

        DSEL_Cursor<String, Integer> filteredBySecond = multiElementCursor
                .fl(join -> join.second() < 30);
        assertEquals(Collections.singletonList(
                cj("Alice", 25)
        ), filteredBySecond.collect());
    }

    @Test
    void testCollect() {
        DSEL_Cursor<String, Integer> multiElementCursor = new ListBackedCursor<>(Arrays.asList(
                cj("Alice", 25),
                cj("Bob", 30),
                cj("Charlie", 35)
        ));

        List<Join<String, Integer>> collected = multiElementCursor.collect();
        assertEquals(3, collected.size());
        assertEquals(cj("Alice", 25), collected.get(0));
    }

    @Test
    void testCount() {
        assertEquals(0, new ListBackedCursor<>(Collections.emptyList()).count());
        assertEquals(1, new ListBackedCursor<>(Collections.singletonList(cj("A", 1))).count());
        assertEquals(3, new ListBackedCursor<>(Arrays.asList(cj("A", 1), cj("B", 2), cj("C", 3))).count());
    }

    @Test
    void testIsEmpty() {
        assertTrue(new ListBackedCursor<>(Collections.emptyList()).isEmp());
        assertFalse(new ListBackedCursor<>(Collections.singletonList(cj("A", 1))).isEmp());
    }

    @Test
    void testFirstJoin() {
        DSEL_Cursor<String, Integer> singleElementCursor = new ListBackedCursor<>(Collections.singletonList(
                cj("A", 1)
        ));
        assertTrue(singleElementCursor.fstJ().isPresent());
        assertEquals(cj("A", 1), singleElementCursor.fstJ().get());

        DSEL_Cursor<String, Integer> emptyCursor = new ListBackedCursor<>(Collections.emptyList());
        assertFalse(emptyCursor.fstJ().isPresent());
    }
}
