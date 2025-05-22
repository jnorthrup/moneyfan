package com.moneyfan.core;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class Pai2Test {

    @Test
    void factoryCreatesPair() {
        Pai2<Integer, String> pair = Pai2.of(1, "one");
        assertEquals(1, pair.first());
        assertEquals("one", pair.second());
    }
}