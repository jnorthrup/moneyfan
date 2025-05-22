package com.moneyfan.grid;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;

class Vect0rTest {

    @Test
    void ofFactoryAndGet() {
        Vect0r<Integer> vec = Vect0r.of(1, 2, 3);
        assertEquals(3, vec.size());
        assertEquals(2, vec.get(1));
    }

    @Test
    void fromList() {
        Vect0r<String> vec = Vect0r.fromList(List.of("a", "b"));
        assertEquals("b", vec.get(1));
    }

    @Test
    void mapIsLazy() {
        AtomicInteger counter = new AtomicInteger();
        Vect0r<Integer> src = new Vect0r<>(4, i -> i);
        Vect0r<Integer> mapped = src.map(Integer.class, val -> {
            counter.incrementAndGet();
            return val * 2;
        });
        // nothing evaluated yet
        assertEquals(0, counter.get());
        assertEquals(6, mapped.get(3)); // triggers one evaluation
        assertEquals(1, counter.get());
    }
}