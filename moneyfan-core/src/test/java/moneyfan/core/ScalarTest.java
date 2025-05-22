package moneyfan.core;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class ScalarTest {
    @Test
    void testConstructionAndAccessors() {
        Scalar scalar = new Scalar(IOMemento.IO_INT, "foo");
        assertEquals(IOMemento.IO_INT, scalar.type());
        assertEquals("foo", scalar.name());
    }

    @Test
    void testImmutability() {
        Scalar scalar = new Scalar(IOMemento.IO_DOUBLE, "bar");
        assertEquals(IOMemento.IO_DOUBLE, scalar.type());
        assertEquals("bar", scalar.name());
    }
}