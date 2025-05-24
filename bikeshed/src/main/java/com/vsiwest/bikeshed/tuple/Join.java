package com.vsiwest.bikeshed.tuple;

import org.jetbrains.annotations.NotNull;
import java.util.Objects;
import java.util.function.BiFunction;
import java.util.function.Function;

public record Join<F, S>(F first, S second) {

    public static <F, S> @NotNull Join<F, S> of(F f, S s) {
        return new Join<>(f, s);
    }

    public @NotNull F fst() {
        return first;
    }

    public @NotNull S snd() {
        return second;
    }

    public <R> @NotNull Join<R, S> mapFst(@NotNull Function<? super F, ? extends R> mapper) {
        Objects.requireNonNull(mapper, "mapper must not be null");
        return new Join<>(mapper.apply(first), second);
    }

    public <R> @NotNull Join<F, R> mapSnd(@NotNull Function<? super S, ? extends R> mapper) {
        Objects.requireNonNull(mapper, "mapper must not be null");
        return new Join<>(first, mapper.apply(second));
    }

    public <R1, R2> @NotNull Join<R1, R2> mapBoth(@NotNull Function<? super F, ? extends R1> fstMapper, @NotNull Function<? super S, ? extends R2> sndMapper) {
        Objects.requireNonNull(fstMapper, "fstMapper must not be null");
        Objects.requireNonNull(sndMapper, "sndMapper must not be null");
        return new Join<>(fstMapper.apply(first), sndMapper.apply(second));
    }

    public <R> @NotNull R map(@NotNull BiFunction<? super F, ? super S, ? extends R> mapper) {
        Objects.requireNonNull(mapper, "mapper must not be null");
        return mapper.apply(first, second);
    }

    public @NotNull Join<S, F> swap() {
        return new Join<>(second, first);
    }

    // ImmutableJoin for internal use or specific scenarios where immutability is enforced
    public static class ImmutableJoin<F, S> extends Join<F, S> {
        protected ImmutableJoin(F first, S second) { // Changed to protected for inheritance
            super(first, second);
        }
    }
}
