package com.vsiwest.bikeshed.type;

import com.vsiwest.bikeshed.core.Join;
import org.jetbrains.annotations.NotNull;

/**
 * Represents metadata for a column, which is a {@link Join} of the column name (String)
 * and its {@link TypeMemento}.
 *
 * In Kotlin: `typealias ColumnMeta = Join<String, TypeMemento>`
 */
public interface ColumnMeta extends Join<String, TypeMemento> {

    /**
     * Factory method to create a ColumnMeta instance.
     * @param name The name of the column.
     * @param type The TypeMemento describing the column's data type.
     * @return A new ColumnMeta instance.
     */
    static @NotNull ColumnMeta of(@NotNull String name, @NotNull TypeMemento type) {
        return new ImmutableColumnMeta(name, type);
    }

    /**
     * Returns the name of the column.
     * @return The column name.
     */
    default @NotNull String name() {
        return fst();
    }

    /**
     * Returns the TypeMemento of the column.
     * @return The column's TypeMemento.
     */
    default @NotNull TypeMemento type() {
        return snd();
    }

    // Inner class for the immutable implementation
    final class ImmutableColumnMeta extends Join.ImmutableJoin<String, TypeMemento> implements ColumnMeta {
        private ImmutableColumnMeta(String name, TypeMemento type) {
            super(name, type);
        }
    }
}
