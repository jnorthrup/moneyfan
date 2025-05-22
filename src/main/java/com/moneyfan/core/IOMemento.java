package com.moneyfan.core;

/**
 * Metadata enum describing IO characteristics for primitive and common data types.
 */
public enum IOMemento {
    IO_INT(4),
    IO_LONG(8),
    IO_DOUBLE(8),
    IO_LOCAL_DATE(4),
    IO_INSTANT(8),
    IO_STRING_FIXED(-1);

    private final int fixedSize;

    IOMemento(int fixedSize) {
        this.fixedSize = fixedSize;
    }

    /**
     * Returns the fixed size in bytes for this IO type or -1 if variable.
     */
    public int fixedSize() {
        return fixedSize;
    }
}