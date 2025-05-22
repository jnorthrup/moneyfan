package com.moneyfan.grid;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class Vect0rTest {

    @Test
    void getReturnsCorrectElement() {
        Vect0r<Integer> vec = Vect0r.of(3, i -> i * 2);
        assertEquals(0, vec.get(0));
        assertEquals(2, vec.get(1));
        assertEquals(4, vec.get(2));
    }

    @Test
    void iteratorIteratesAllElements() {
        Vect0r<String> vec = Vect0r.fromList(java.util.List.of("a", "b", "c"));
        StringBuilder sb = new StringBuilder();
        for (String s : vec) sb.append(s);
        assertEquals("abc", sb.toString());
    }
}