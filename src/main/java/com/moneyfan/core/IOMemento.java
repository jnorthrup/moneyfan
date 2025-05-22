package com.moneyfan.core;

/**
 * Enum describing supported IO types and their fixed-size characteristics.
 */
public enum IOMemento {
    IO_INT(4),
    IO_LONG(8),
    IO_DOUBLE(8),
    IO_LOCAL_DATE(4), // stores epoch day as int
    IO_INSTANT(12),   // stores epochSecond (long) + nano (int)
    IO_STRING_FIXED(-1); // variable depending on metadata

    private final int fixedSize;

    IOMemento(int fixedSize) {
        this.fixedSize = fixedSize;
    }

    /**
     * Returns the fixed size in bytes, or -1 if variable.
     */
    public int fixedSize() {
        return fixedSize;
    }

    /**
     * @return true if the type has known fixed size in bytes.
     */
    public boolean isFixedSize() {
        return fixedSize > 0;
    }
}