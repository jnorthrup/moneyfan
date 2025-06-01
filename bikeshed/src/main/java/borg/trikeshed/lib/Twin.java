package borg.trikeshed.lib; // Changed package

// No explicit import for borg.trikeshed.lib.Join needed if Twin is in the same package.
// If Join were in a different sub-package of borg.trikeshed.lib, an import would be needed.
import org.jetbrains.annotations.NotNull;

/**
 * A specialized {@link Join} for symmetric pairs, where both elements are of the same type.
 *
 * @param <T> The common type of both elements.
 */
public interface Twin<T> extends Join<T, T> {

    /**
     * Factory method to create a new Twin instance.
     *
     * @param fst The first element.
     * @param snd The second element.
     * @param <T> The type of elements.
     * @return A new immutable Twin instance.
     */
    static <T> @NotNull Twin<T> of(T fst, T snd) {
        return new ImmutableTwin<>(fst, snd);
    }

    // Inner class for the immutable implementation
    final class ImmutableTwin<T> implements Twin<T> { // Removed "extends Join.ImmutableJoin<T, T>"
        private final T first_element;
        private final T second_element;

        private ImmutableTwin(T fst, T snd) {
            this.first_element = java.util.Objects.requireNonNull(fst, "first element must not be null");
            this.second_element = java.util.Objects.requireNonNull(snd, "second element must not be null");
        }

        @Override
        public T fst() {
            return first_element;
        }

        @Override
        public T snd() {
            return second_element;
        }
    }
}
