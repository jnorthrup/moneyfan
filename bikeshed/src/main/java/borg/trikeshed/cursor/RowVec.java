package borg.trikeshed.cursor; // Changed package

import borg.trikeshed.isam.RecordMeta; // New import for ColumnMeta replacement
import borg.trikeshed.lib.Join;       // New import
import borg.trikeshed.lib.Series;      // New import
// Original com.yourdomain.bikeshed.types.ColumnMeta will be unused

import java.util.Objects;
import java.util.function.Function;
import java.util.function.IntFunction;
import java.util.function.Supplier;

public class RowVec extends borg.trikeshed.lib.Series<borg.trikeshed.lib.Join<Object, java.util.function.Supplier<borg.trikeshed.isam.RecordMeta>>> {
    public RowVec(int size, IntFunction<borg.trikeshed.lib.Join<Object, java.util.function.Supplier<borg.trikeshed.isam.RecordMeta>>> provider) {
        super(size, provider);
    }

    @SafeVarargs
    public static RowVec of(borg.trikeshed.lib.Join<Object, java.util.function.Supplier<borg.trikeshed.isam.RecordMeta>>... valuesAndMeta) {
        Objects.requireNonNull(valuesAndMeta, "valuesAndMeta array must not be null");
        return new RowVec(valuesAndMeta.length, i -> valuesAndMeta[i]);
    }

    public Object getValue(int index) {
        return get(index).first();
    }

    public borg.trikeshed.isam.RecordMeta getMeta(int index) {
        return get(index).second().get();
    }

    public <R> Series<R> mapValues(Function<Object, R> mapper) { // Series here is borg.trikeshed.lib.Series due to import
        Objects.requireNonNull(mapper, "mapper must not be null");
        return this.map(join -> mapper.apply(join.first()));
    }
}
