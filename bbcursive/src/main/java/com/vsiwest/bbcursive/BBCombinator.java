package com.vsiwest.bbcursive;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.function.Function;

public enum BBCombinator { INSTANCE;
    @SafeVarargs public static <T> Cursive<List<T>> sequence(Cursive<? extends T>... parsers) {
        Objects.requireNonNull(parsers); return buffer -> { int pos = buffer.position(); List<T> res = new ArrayList<>(parsers.length); for (Cursive<? extends T> p : parsers) { T r = p.apply(buffer); if (r == null) { buffer.position(pos); return null; } res.add(r); } return res; };
    }
    @SafeVarargs public static <T> Cursive<T> choice(Cursive<? extends T>... parsers) {
        Objects.requireNonNull(parsers); return buffer -> { for (Cursive<? extends T> p : parsers) { int pos = buffer.position(); T r = p.apply(buffer); if (r != null) return r; buffer.position(pos); } return null; };
    }
    public static <T> Cursive<Optional<T>> optional(Cursive<T> parser) {
        Objects.requireNonNull(parser); return buffer -> { int pos = buffer.position(); T r = parser.apply(buffer); if (r != null) return Optional.of(r); buffer.position(pos); return Optional.empty(); };
    }
    public static <T> Cursive<List<T>> many(Cursive<T> parser) {
        Objects.requireNonNull(parser); return buffer -> { List<T> res = new ArrayList<>(); while (true) { int pos = buffer.position(); T r = parser.apply(buffer); if (r != null) res.add(r); else { buffer.position(pos); break; } } return res; };
    }
    public static <T> Cursive<List<T>> many1(Cursive<T> parser) {
        Objects.requireNonNull(parser); return buffer -> { List<T> r = many(parser).apply(buffer); return (r != null && !r.isEmpty()) ? r : null; };
    }
    public static <T, U> Cursive<T> terminated(Cursive<T> parser, Cursive<U> terminator) {
        Objects.requireNonNull(parser); Objects.requireNonNull(terminator); return buffer -> { int pos = buffer.position(); T r = parser.apply(buffer); if (r != null) { U termR = terminator.apply(buffer); if (termR != null) return r; } buffer.position(pos); return null; };
    }
    public static <T, S> Cursive<List<T>> sepBy(Cursive<T> item, Cursive<S> sep) {
        Objects.requireNonNull(item); Objects.requireNonNull(sep); return buffer -> { List<T> res = new ArrayList<>(); T first = item.apply(buffer); if (first == null) return res; res.add(first); while (true) { int pos = buffer.position(); S sR = sep.apply(buffer); if (sR != null) { T nR = item.apply(buffer); if (nR != null) res.add(nR); else { buffer.position(pos); break; } } else { buffer.position(pos); break; } } return res; };
    }
    public static <T, S> Cursive<List<T>> sepBy1(Cursive<T> item, Cursive<S> sep) {
        Objects.requireNonNull(item); Objects.requireNonNull(sep); return buffer -> { List<T> r = sepBy(item, sep).apply(buffer); return (r != null && !r.isEmpty()) ? r : null; };
    }
    public static <T, R> Cursive<R> map(Cursive<T> parser, Function<T, R> fn) {
        Objects.requireNonNull(parser); Objects.requireNonNull(fn); return buffer -> { T r = parser.apply(buffer); return r != null ? fn.apply(r) : null; };
    }
}
