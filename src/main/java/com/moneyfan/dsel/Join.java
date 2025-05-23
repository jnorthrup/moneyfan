/**
 * An immutable 2-ary tuple record.
 *
 * @param <F> the type of the first element
 * @param <S> the type of the second element
 */
public record Join<F, S>(F first, S second) {

    /**
     * Swaps the elements of the tuple.
     * @return a new Join with the elements swapped.
     */
    public Join<S, F> swap() {
        return new Join<>(second, first);
    }

    public static <FT, ST> Join<FT, ST> j(FT first, ST second) { return new Join<>(first, second); }
}
