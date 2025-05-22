package io.moneyfan.core;

import org.junit.jupiter.api.Test;

import java.util.OptionalInt;

import static org.junit.jupiter.api.Assertions.*;

class IOMementoTest {

    @Test
    void varcharHasStringRuntimeClass() {
        assertEquals(String.class, IOMemento.VARCHAR.runtimeClass(),
                "VARCHAR should map to String.class at runtime");
    }

    @Test
    void byteArrayHasByteArrayRuntimeClass() {
        assertEquals(byte[].class, IOMemento.BYTE_ARRAY.runtimeClass(),
                "BYTE_ARRAY should map to byte[].class at runtime");
    }

    @Test
    void fixedSizeTypesReportCorrectSizes() {
        assertFalse(IOMemento.INT.isVariableLength());
        assertEquals(OptionalInt.of(4), IOMemento.INT.fixedSizeBytes());

        assertFalse(IOMemento.DOUBLE.isVariableLength());
        assertEquals(OptionalInt.of(8), IOMemento.DOUBLE.fixedSizeBytes());
    }

    @Test
    void variableLengthTypesReportVariable() {
        assertTrue(IOMemento.VARCHAR.isVariableLength());
        assertTrue(IOMemento.BYTE_ARRAY.isVariableLength());

        assertEquals(OptionalInt.empty(), IOMemento.VARCHAR.fixedSizeBytes());
        assertEquals(OptionalInt.empty(), IOMemento.BYTE_ARRAY.fixedSizeBytes());
    }
}