package com.vsiwest.moneyfan.bikeshed.core;

import java.util.Objects;
import java.util.function.BiFunction;
import java.util.function.Function;

public record Join<F, S>(F first, S second) {

    public static <F, S> Join<F, S> of(F f, S s) {
        return new Join<>(f, s);
    }

    public <R> Join<R, S> mapFst(Function<? super F, ? extends R> mapper) {
        Objects.requireNonNull(mapper, "mapper must not be null");
        return new Join<>(mapper.apply(first), second);
    }

    public <R> Join<F, R> mapSnd(Function<? super S, ? extends R> mapper) {
        Objects.requireNonNull(mapper, "mapper must not be null");
        return new Join<>(first, mapper.apply(second));
    }

    public <R1, R2> Join<R1, R2> mapBoth(Function<? super F, ? extends R1> fstMapper, Function<? super S, ? extends R2> sndMapper) {
        Objects.requireNonNull(fstMapper, "fstMapper must not be null");
        Objects.requireNonNull(sndMapper, "sndMapper must not be null");
        return new Join<>(fstMapper.apply(first), sndMapper.apply(second));
    }

    public Join<S, F> swap() {
        return new Join<>(second, first);
    }

    public <F2, S2> Join<Join<F, S>, Join<F2, S2>> plus(Join<F2, S2> other) {
        Objects.requireNonNull(other, "other Join must not be null");
        return Join.of(this, other);
    }

    public Object get(int index) {
        if (index == 0) return first;
        if (index == 1) return second;
        throw new IndexOutOfBoundsException("Index " + index + " out of bounds for Join (0-1)");
    }

    public <R> R map(BiFunction<? super F, ? super S, ? extends R> mapper) {
        Objects.requireNonNull(mapper, "mapper must not be null");
        return mapper.apply(first, second);
    }

    // ImmutableJoin for internal use or specific scenarios where immutability is enforced
    public static class ImmutableJoin<F, S> extends Join<F, S> {
        private ImmutableJoin(F first, S second) {
            super(first, second); // Call record constructor
        }
    }
}
