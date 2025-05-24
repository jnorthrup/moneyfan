package com.vsiwest.bikeshed.dsel;

import com.vsiwest.bikeshed.core.Join;
import com.vsiwest.bikeshed.core.Series;
import com.vsiwest.bikeshed.core.RowVec;
import com.vsiwest.bikeshed.core.Twin;
import com.vsiwest.bikeshed.types.ColumnMeta;
import com.vsiwest.bikeshed.types.IOMemento;

import java.util.List;
import java.util.Objects;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.function.Supplier;

/**
 * D - The Domain-Specific Embedded Language (DSEL) for data processing.
 * This enum provides static-like methods that act as global DSEL operations,
 * offering a concise, "glyph-based shorthand" style for common data manipulations.
 * It emphasizes immutability, compositional purity, and function references.
 */
public enum D {
    // This enum doesn't need instances; it's a namespace for utility methods.
    ; // No enum constants

    public static final String META_SFX = ".meta";

    // --- Join Operations (Glyphs: jn, fst, snd, mapBoth, swap, test) ---

    /**
     * Shorthand for {@code Join.of(f, s)}. Creates an immutable 2-tuple.
     * Glyph: `jn`
     * Example: `D.jn("hello", 123)`
     *
     * @param f The first element.
     * @param s The second element.
     * @param <F> Type of the first element.
     * @param <S> Type of the second element.
     * @return A new Join instance.
     */
    public static <F, S> @NotNull Join<F, S> jn(F f, S s) {
        return Join.of(f, s);
    }

    /**
     * Transforms the first element of a Join. Shorthand for {@code join.mapFst(mapper)}.
     * Glyph: `fst` (read as "first")
     * Example: `D.fst(D.jn("abc", 123), s -> s.toUpperCase())`
     *
     * @param join The original Join.
     * @param mapper The function to apply to the first element.
     * @param <F> Original type of the first element.
     *