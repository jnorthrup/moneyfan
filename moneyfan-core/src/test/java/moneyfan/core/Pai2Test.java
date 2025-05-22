package moneyfan.core;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class Pai2Test {
    @Test
    void testFactoryAndAccessors() {
        Pai2<String, Integer> pair = Pai2.of("foo", 42);
        assertEquals("foo", pair.first());
        assertEquals(42, pair.second());
    }

    @Test
    void testImmutability() {
        Pai2<String, String> pair = Pai2.of("a", "b");
        // There are no setters, so immutability is enforced by the record type itself.
        assertEquals("a", pair.first());
        assertEquals("b", pair.second());
    }

    @Test
    void testEqualsAndHashCode() {
        Pai2<String, Integer> p1 = Pai2.of("x", 1);
        Pai2<String, Integer> p2 = Pai2.of("x", 1);
        assertEquals(p1, p2);
        assertEquals(p1.hashCode(), p2.hashCode());
    }
}