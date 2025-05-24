package com.vsiwest.bbcursive.core;

import org.jetbrains.annotations.Nullable;
import java.util.Objects;

public record Join<F, S>(@Nullable F first, @Nullable S second) {
    public @Nullable F fst() { return first; }
    public @Nullable S snd() { return second; }
    @Override public boolean equals(Object o) { if (this == o) return true; if (o == null || getClass() != o.getClass()) return false; Join<?, ?> join = (Join<?, ?>) o; return Objects.equals(first, join.first) && Objects.equals(second, join.second); }
    @Override public int hashCode() { return Objects.hash(first, second); }
}
