package com.moneyfan.core;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class Pai2Test {
    @Test
    void factoryAndAccessorsWork() {
        Pai2<String, Integer> pair = Pai2.of("foo", 42);
        assertEquals("foo", pair.first());
        assertEquals(42, pair.second());
    }

    @Test
    void mapFirstSecond() {
        Pai2<String, Integer> pair = Pai2.of("foo", 1);
        Pai2<Integer, Integer> mappedFirst = pair.mapFirst(String::length);
        assertEquals(3, mappedFirst.first());
        assertEquals(1, mappedFirst.second());

        Pai2<String, String> mappedSecond = pair.mapSecond(Object::toString);
        assertEquals("foo", mappedSecond.first());
        assertEquals("1", mappedSecond.second());
    }
}