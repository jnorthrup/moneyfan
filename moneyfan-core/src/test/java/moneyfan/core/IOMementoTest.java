package moneyfan.core;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class IOMementoTest {
    @Test
    void testEnumValues() {
        assertEquals(Integer.class, IOMemento.IO_INT.javaClass());
        assertEquals(4, IOMemento.IO_INT.fixedSize());
        assertTrue(IOMemento.IO_INT.isFixedSize());

        assertEquals(Long.class, IOMemento.IO_LONG.javaClass());
        assertEquals(8, IOMemento.IO_LONG.fixedSize());
        assertTrue(IOMemento.IO_LONG.isFixedSize());

        assertEquals(Double.class, IOMemento.IO_DOUBLE.javaClass());
        assertEquals(8, IOMemento.IO_DOUBLE.fixedSize());
        assertTrue(IOMemento.IO_DOUBLE.isFixedSize());

        assertEquals(java.time.LocalDate.class, IOMemento.IO_LOCAL_DATE.javaClass());
        assertEquals(4, IOMemento.IO_LOCAL_DATE.fixedSize());
        assertTrue(IOMemento.IO_LOCAL_DATE.isFixedSize());

        assertEquals(java.time.Instant.class, IOMemento.IO_INSTANT.javaClass());
        assertEquals(12, IOMemento.IO_INSTANT.fixedSize());
        assertTrue(IOMemento.IO_INSTANT.isFixedSize());

        assertEquals(String.class, IOMemento.IO_STRING_FIXED.javaClass());
        assertEquals(-1, IOMemento.IO_STRING_FIXED.fixedSize());
        assertFalse(IOMemento.IO_STRING_FIXED.isFixedSize());
    }
}