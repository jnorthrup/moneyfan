/**
 * A simple immutable 2-ary tuple (Pair).
 * This is the core data structure for our DSEL.
 *
 * @param <F> Type of the first element.
 * @param <S> Type of the second element.
 */
public record Join<F, S>(F first, S second) {}
