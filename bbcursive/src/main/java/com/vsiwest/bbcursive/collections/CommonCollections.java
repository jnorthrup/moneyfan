package com.vsiwest.bbcursive.collections;

import com.vsiwest.bbcursive.core.Join;
import com.vsiwest.bbcursive.core.Series;
import org.jetbrains.annotations.Nullable;

public interface Twin<T> extends Join<T, T> {}
record TwinImpl<T>(@Nullable T first, @Nullable T second) implements Twin<T>{}

public interface Either<L,R> {
    default @Nullable L leftOrNull() { return this instanceof Left<L,R> l ? l.value() : null; }
    default @Nullable R rightOrNull() { return this instanceof Right<L,R> r ? r.value() : null; }
    default boolean isLeft() { return this instanceof Left; }
    default boolean isRight() { return this instanceof Right; }
}
record Left<L,R>(@Nullable L value) implements Either<L,R> {}
record Right<L,R>(@Nullable R value) implements Either<L,R> {}

public interface Bucket<T> extends Series<T> {}
