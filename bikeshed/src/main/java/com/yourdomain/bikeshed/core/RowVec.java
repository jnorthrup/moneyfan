package com.yourdomain.bikeshed.core;

import com.yourdomain.bikeshed.type.ColumnMeta;
import org.jetbrains.annotations.NotNull;

import java.util.List;
import java.util.function.Supplier;

/**
 * Represents a row of data, where each element is a {@link Join} of a value and a
 * {@link Supplier} for its {@link ColumnMeta}. This allows for lazy retrieval of metadata.
 *
 * In Kotlin: `typealias RowVec = Series<Join<Any?, () -> ColumnMeta>>`
 */
public interface RowVec extends Series<Join<Object, Supplier<ColumnMeta>>> {

    /**
     * Factory method to create a RowVec from a list of value-metadata pairs.
     * @param valuesAndMeta A list of Join instances, each containing a value and its ColumnMeta supplier.
     * @return A new RowVec instance.
     */
    static @NotNull RowVec of(@NotNull List<Join<Object, Supplier<ColumnMeta>>> valuesAndMeta) {
        return Series.of(valuesAndMeta.size(), valuesAndMeta::get);
    }
}
