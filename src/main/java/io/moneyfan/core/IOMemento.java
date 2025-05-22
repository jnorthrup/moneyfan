package io.moneyfan.core;

import java.util.OptionalInt;

/**
 * IOMemento enumerates the primitive and compound value kinds that can be stored
 * in a Moneyfan column. Each constant carries basic metadata that can be used at
 * runtime (e.g. by cell drivers) without resorting to reflection-heavy logic.
 */
public enum IOMemento {

    // Fixed-width primitives
    INT(Integer.class, 4),
    LONG(Long.class, 8),
    DOUBLE(Double.class, 8),

    // Java time primitives (encoded as epoch seconds / days, so still fixed-width)
    LOCAL_DATE(java.time.LocalDate.class, 4),
    INSTANT(java.time.Instant.class, 12),

    // Variable-width primitives
    VARCHAR(String.class, -1),
    BYTE_ARRAY(byte[].class, -1);

    private final Class<?> runtimeClass;
    private final int fixedSizeBytes; // -1 denotes variable length.

    IOMemento(Class<?> runtimeClass, int fixedSizeBytes) {
        this.runtimeClass = runtimeClass;
        this.fixedSizeBytes = fixedSizeBytes;
    }

    /**
     * Returns the JVM class representing values of this memento.
     */
    public Class<?> runtimeClass() {
        return runtimeClass;
    }

    /**
     * Returns true if this memento has variable-width encoding.
     */
    public boolean isVariableLength() {
        return fixedSizeBytes < 0;
    }

    /**
     * Returns the fixed size in bytes if applicable.
     */
    public OptionalInt fixedSizeBytes() {
        return fixedSizeBytes < 0 ? OptionalInt.empty() : OptionalInt.of(fixedSizeBytes);
    }
}