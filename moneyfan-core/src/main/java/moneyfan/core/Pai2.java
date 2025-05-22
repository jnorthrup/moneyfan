package moneyfan.core;

import java.util.Objects;

public record Pai2<F, S>(F first, S second) {
    public static <F, S> Pai2<F, S> of(F first, S second) {
        return new Pai2<>(first, second);
    }
    
    // Optionally, override equals/hashCode/toString if needed, but record provides them by default.
}