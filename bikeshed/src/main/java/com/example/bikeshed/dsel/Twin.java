package com.example.bikeshed.dsel;

import com.example.bikeshed.core.Join;

/**
 * A specialized `Join<T, T>` for symmetric pairs.
 * This is effectively a type alias for {@link com.example.bikeshed.core.Join} where both types are the same.
 */
public class Twin<T> extends Join<T, T> {

    protected Twin(T first, T second) {
        super(first, second);
    }

    /**
     * Factory method for creating a Twin instance.
     * @param first The first element.
     * @param second The second element.
     * @param <T> The type of elements.
     * @return A new immutable Twin instance.
     */
    public static <T> Twin<T> of(T first, T second) {
        return new Twin<>(first, second);
    }
}
