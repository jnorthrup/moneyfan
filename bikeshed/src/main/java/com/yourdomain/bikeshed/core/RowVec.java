package com.yourdomain.bikeshed.core;

import com.yourdomain.bikeshed.types.ColumnMeta;
import java.util.Objects;
import java.util.function.Function;
import java.util.function.IntFunction;
import java.util.function.Supplier;

public class RowVec extends Series<Join<Object, Supplier<ColumnMeta>>> {
    public RowVec(int size, IntFunction<Join<Object, Supplier<ColumnMeta>>> provider) {
        super(size, provider);
    }

    @SafeVarargs
    public static RowVec of(Join<Object, Supplier<ColumnMeta>>... valuesAndMeta) {
        Objects.requireNonNull(valuesAndMeta, "valuesAndMeta array must not be null");
        return new RowVec(valuesAndMeta.length, i -> valuesAndMeta[i]);
    }

    public Object getValue(int index) {
        return get(index).first();
    }

    public ColumnMeta getMeta(int index) {
        return get(index).second().get();
    }

    public <R> Series<R> mapValues(Function<Object, R> mapper) {
        Objects.requireNonNull(mapper, "mapper must not be null");
        return this.map(join -> mapper.apply(join.first()));
    }
}
