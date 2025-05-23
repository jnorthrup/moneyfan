package com.moneyfan.dsel.core;

import java.util.List;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.stream.Collectors;

/**
 * Represents a Cursor over a collection of {@link RowVec} instances.
 * This is essentially a {@link Series} of {@link RowVec}, providing a table-like
 * structure for DSEL operations.
 */
public record Cursor(Series<RowVec> series) {

    /**
     * Static factory method for creating a Cursor from a {@link Series} of {@link RowVec}.
     */
    public static Cursor csr(Series<RowVec> series) {
        return new Cursor(series);
    }

    /**
     * Static factory method for creating a Cursor from a {@link Series} of {@link RowVec}.
     */
    public static Cursor fromSeries(Series<RowVec> rows) {
        return new Cursor(rows);
    }

    

    /**
     * Retrieves the {@link RowVec} at the specified 0-based index.
     */
    public RowVec getRow(int index) {
        return series.at(index);
    }

    /**
     * Applies a mapping function to each {@link RowVec} in the Cursor, producing a new Cursor.
     * The mapper function must return a {@link RowVec}.
     *
     * @param rowMapper A function to transform each {@link RowVec}.
     * @return A new Cursor with transformed rows.
     */
    public Cursor map(Function<? super RowVec, ? extends RowVec> rowMapper) {
        return new Cursor(series.mapVal(rowMapper));
    }

    /**
     * Filters the Cursor based on a predicate applied to each {@link RowVec}.
     *
     * @param rowPredicate A predicate to test each {@link RowVec}.
     * @return A new Cursor containing only the rows that satisfy the predicate.
     */
    public Cursor filter(Predicate<? super RowVec> rowPredicate) {
        return new Cursor(series.filterVal(rowPredicate));
    }
}