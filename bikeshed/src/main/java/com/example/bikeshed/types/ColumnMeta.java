package com.example.bikeshed.types;

import com.example.bikeshed.dsel.Join;

/**
 * Type alias for `ColumnMeta` as `Join<String, TypeMemento>`.
 * Encapsulates a column's name and its type information.
 */
public class ColumnMeta extends Join<String, TypeMemento> {

    protected ColumnMeta(String name, TypeMemento type) {
        super(name, type);
    }

    /**
     * Factory method for ColumnMeta.
     * @param name The name of the column.
     * @param type The TypeMemento representing the column's data type.
     * @return A new ColumnMeta instance.
     */
    public static ColumnMeta of(String name, TypeMemento type) {
        return new ColumnMeta(name, type);
    }

    /**
     * Mix-in for name (accessed as `a` in `Join`).
     * @return The name of the column.
     */
    public String getName() {
        return a();
    }

    /**
     * Mix-in for type (accessed as `b` in `Join`).
     * @return The TypeMemento of the column.
     */
    public TypeMemento getType() {
        return b();
    }
}
