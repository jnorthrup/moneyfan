package com.vsiwest.moneyfan.bikeshed.core;

import java.util.Objects;
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

    public <R1, R2> Join<R1, R2> mapBoth(Function<? super F, ? extends R1> mapperFst, Function<? super S, ? extends R2> mapperSnd) {
        Objects.requireNonNull(mapperFst, "mapperFst must not be null");
        Objects.requireNonNull(mapperSnd, "mapperSnd must not be null");
        return new Join<>(mapperFst.apply(first), mapperSnd.apply(second));
    }

    public Join<S, F> swap() {
        return new Join<>(second, first);
    }

    public Object get(int index) {
        if (index == 0) return first;
        if (index == 1) return second;
        throw new IndexOutOfBoundsException("Index " + index + " out of bounds for Join (0-1)");
    }

    // ImmutableJoin for internal use or specific scenarios where immutability is enforced
    public static class ImmutableJoin<F, S> extends Join<F, S> {
        private ImmutableJoin(F first, S second) {
            super(first, second); // Call record constructor
            this.immutableFirst = first;
            this.immutableSecond = second;
        }

        private final F immutableFirst;
        private final S immutableSecond;

        @Override
        public F first() {
            return immutableFirst;
        }

        @Override
        public S second() {
            return immutableSecond;
        }

        // Override map methods to return ImmutableJoin if desired, or just rely on super's behavior
        @Override
        public <R> Join<R, S> mapFst(Function<? super F, ? extends R> mapper) {
            return new ImmutableJoin<>(mapper.apply(immutableFirst), immutableSecond);
        }

        @Override
        public <R> Join<F, R> mapSnd(Function<? super S, ? extends R> mapper) {
            return new ImmutableJoin<>(immutableFirst, mapper.apply(immutableSecond));
        }

        @Override
        public <R1, R2> Join<R1, R2> mapBoth(Function<? super F, ? extends R1> mapperFst, Function<? super S, ? extends R2> mapperSnd) {
            return new ImmutableJoin<>(mapperFst.apply(immutableFirst), mapperSnd.apply(immutableSecond));
        }

        @Override
        public Join<S, F> swap() {
            return new ImmutableJoin<>(immutableSecond, immutableFirst);
        }
    }
}
