package com.vsiwest.bikeshed.types;

import com.vsiwest.bikeshed.tuple.Join;
import org.jetbrains.annotations.NotNull;
import java.util.Objects;

/**
 * Represents metadata for a single column, including its name and data type.
 * Implements {@link Join} to pair the column name (String) with its type (TypeMemento).
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
        return fst();
    }

    /**
     * Returns the TypeMemento describing the column's data type.
     * @return The column's TypeMemento.
     */
    default @NotNull TypeMemento type() {
        return snd();
    }

    /**
     * Immutable implementation of the ColumnMeta interface.
     */
    final class ImmutableColumnMeta extends Join.ImmutableJoin<String, TypeMemento> implements ColumnMeta {
        private ImmutableColumnMeta(String name, TypeMemento type) {
            super(name, type);
        }

        @Override
        public @NotNull String name() {
            return first();
        }

        @Override
        public @NotNull TypeMemento type() {
            return second();
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            ImmutableColumnMeta that = (ImmutableColumnMeta) o;
            return Objects.equals(name(), that.name()) &&
                   Objects.equals(type(), that.type());
        }

        @Override
        public int hashCode() {
            return Objects.hash(name(), type());
        }

        @Override
        public String toString() {
            return "ColumnMeta{name='" + name() + "', type=" + type() + '}';
        }
    }
}
