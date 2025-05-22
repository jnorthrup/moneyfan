package moneyfan.grid;

import org.junit.jupiter.api.Test;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import static org.junit.jupiter.api.Assertions.*;

class Vect0rTest {
    @Test
    void testOfAndGet() {
        Vect0r<String> v = Vect0r.of(3, i -> "val" + i);
        assertEquals(3, v.size());
        assertEquals("val0", v.get(0));
        assertEquals("val1", v.get(1));
        assertEquals("val2", v.get(2));
        assertThrows(IndexOutOfBoundsException.class, () -> v.get(3));
    }

    @Test
    void testFromList() {
        List<Integer> list = List.of(10, 20, 30);
        Vect0r<Integer> v = Vect0r.fromList(list);
        assertEquals(3, v.size());
        assertEquals(10, v.get(0));
        assertEquals(20, v.get(1));
        assertEquals(30, v.get(2));
    }

    @Test
    void testImmutability() {
        Vect0r<String> v = Vect0r.of(2, i -> "x" + i);
        assertEquals("x0", v.get(0));
        assertEquals("x1", v.get(1));
        // No mutator methods, so immutability is enforced by the record type itself.
    }

    @Test
    void testLazyAccess() {
        AtomicInteger counter = new AtomicInteger(0);
        Vect0r<Integer> v = Vect0r.of(2, i -> counter.incrementAndGet());
        assertEquals(1, v.get(0));
        assertEquals(2, v.get(1));
        // Accessing again should increment again (demonstrates laziness, not caching)
        assertEquals(3, v.get(0));
    }
}