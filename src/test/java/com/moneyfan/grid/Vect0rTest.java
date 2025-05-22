package com.moneyfan.grid;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class Vect0rTest {

    @Test
    void ofFactoryAndGetWork() {
        Vect0r<Integer> v = Vect0r.of(3, i -> i + 10);
        assertEquals(3, v.size());
        assertEquals(12, v.get(2));
    }

    @Test
    void fromListCreatesVector() {
        List<String> items = List.of("a", "b", "c");
        Vect0r<String> v = Vect0r.fromList(items);
        assertEquals(3, v.size());
        assertEquals("b", v.get(1));
    }

    @Test
    void getThrowsOnBadIndex() {
        Vect0r<Integer> v = Vect0r.of(1, i -> 0);
        assertThrows(IndexOutOfBoundsException.class, () -> v.get(-1));
        assertThrows(IndexOutOfBoundsException.class, () -> v.get(1));
    }
}