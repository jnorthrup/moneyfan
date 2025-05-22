package moneyfan.core;

public enum IOMemento {
    IO_INT(Integer.class, 4, true),
    IO_LONG(Long.class, 8, true),
    IO_DOUBLE(Double.class, 8, true),
    IO_LOCAL_DATE(java.time.LocalDate.class, 4, true), // e.g., epoch days as int
    IO_INSTANT(java.time.Instant.class, 12, true), // e.g., epoch seconds (8) + nanos (4)
    IO_STRING_FIXED(String.class, -1, false); // size determined by metadata

    private final Class<?> javaClass;
    private final int fixedSize;
    private final boolean isFixedSize;

    IOMemento(Class<?> javaClass, int fixedSize, boolean isFixedSize) {
        this.javaClass = javaClass;
        this.fixedSize = fixedSize;
        this.isFixedSize = isFixedSize;
    }

    public Class<?> javaClass() { return javaClass; }
    public int fixedSize() { return fixedSize; }
    public boolean isFixedSize() { return isFixedSize; }
}