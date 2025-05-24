package com.vsiwest.moneyfan.bikeshed.types;

import com.vsiwest.moneyfan.bikeshed.core.Join;
import org.jetbrains.annotations.NotNull;

import java.util.Objects;

/**
 * Represents metadata for a single column in a tabular data structure.
 * It's a specialized {@link Join} of a column name (String) and its type description (TypeMemento).
 */
public interface ColumnMeta extends Join<String, TypeMemento> {

    /**
     * Factory method to create a ColumnMeta instance.
     * @param name The name of the column.
     * @param type The TypeMemento describing the column's data type.
     * @return A new ColumnMeta instance.
     */
    static @NotNull ColumnMeta of(@NotNull String name, @NotNull TypeMemento type) {
        Objects.requireNonNull(name, "name must not be null");
        Objects.requireNonNull(type, "type must not be null");
        return new ImmutableColumnMeta(name, type);
    }

    /**
     * Returns the name of the column.
     * @return The column name.
     */
    default @NotNull String name() {
        return first();
    }

    /**
     * Returns the TypeMemento describing the column's data type.
     * @return The column's TypeMemento.
     */
    default @NotNull TypeMemento type() {
        return second();
    }

    /**
     * Immutable implementation of ColumnMeta.
     * Uses {@link Join.ImmutableJoin} for its backing.
     */
    final class ImmutableColumnMeta extends Join.ImmutableJoin<String, TypeMemento> implements ColumnMeta {
        private ImmutableColumnMeta(String name, TypeMemento type) {
            super(name, type);
        }
    }
}
