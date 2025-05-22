package com.moneyfan.core;

import java.util.Objects;

/**
 * Defines a column's physical type ({@link IOMemento}) and logical name.
 * <p>
 *     If the underlying type is {@link IOMemento#STRING_FIXED} the {@code fixedLength} field must
 *     be &gt; 0 and represents the number of bytes reserved for the UTF-8 string in the ISAM record.
 * </p>
 */
public record Scalar(IOMemento type, String name, int fixedLength) {

    public Scalar {
        Objects.requireNonNull(type, "type");
        Objects.requireNonNull(name, "name");
        if (type == IOMemento.STRING_FIXED && fixedLength <= 0) {
            throw new IllegalArgumentException("STRING_FIXED requires positive fixedLength");
        }
        if (type != IOMemento.STRING_FIXED && fixedLength != 0) {
            throw new IllegalArgumentException("fixedLength only valid for STRING_FIXED type");
        }
    }

    /**
     * Convenience factory when no fixed length is required.
     */
    public static Scalar of(IOMemento type, String name) {
        return new Scalar(type, name, 0);
    }

    /**
     * Convenience factory for {@code STRING_FIXED} columns.
     */
    public static Scalar fixedString(String name, int length) {
        return new Scalar(IOMemento.STRING_FIXED, name, length);
    }

    public boolean isFixedString() {return type == IOMemento.STRING_FIXED;}
}