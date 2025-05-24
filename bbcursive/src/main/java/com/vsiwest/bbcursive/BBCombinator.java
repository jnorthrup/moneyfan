package com.vsiwest.bbcursive;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.function.Function;
import java.util.function.Predicate;

/**
 * Provides higher-order parsing functions (combinators) to compose `Cursive` parsers.
 * These functions take one or more parsers and return a new parser.
 */
public enum BBCombinator {
    // Enum acts as a namespace for static combinator methods.
    // No instance-specific state or behavior.
    INSTANCE; // Singleton instance

    /**
     * Creates a parser that attempts to apply a sequence of parsers.
     * All parsers must succeed in order for the sequence to succeed.
     *
     * @param parsers An array of Cursive parsers.
     * @return A Cursive parser that returns a List of results from the successful sequence, or null if any fails.
     */
    @SafeVarargs
    public static <T> Cursive<List<T>> sequence(Cursive<T>... parsers) {
        Objects.requireNonNull(parsers, "Parsers array cannot be null");
        return buffer -> {
            int originalPosition = buffer.position();
            List<T> results = new ArrayList<>(parsers.length);
            for (Cursive<T> parser : parsers) {
                T result = parser.apply(buffer);
                if (result == null) {
                    buffer.position(originalPosition); // Rewind on failure
                    return null;
                }
                results.add(result);
            }
            return results;
        };
    }

    /**
     * Creates a parser that attempts to apply the first successful parser from a list.
     *
     * @param parsers An array of Cursive parsers.
     * @return A Cursive parser that returns the result of the first successful parser, or null if all fail.
     */
    @SafeVarargs
    public static <T> Cursive<T> choice(Cursive<T>... parsers) {
        Objects.requireNonNull(parsers, "Parsers array cannot be null");
        return buffer -> {
            for (Cursive<T> parser : parsers) {
                int originalPosition = buffer.position();
                T result = parser.apply(buffer);
                if (result != null) {
                    return result; // First successful parser wins
                }
                buffer.position(originalPosition); // Rewind on failure, try next
            }
            return null; // All parsers failed
        };
    }

    /**
     * Creates a parser that makes another parser optional.
     * If the inner parser succeeds, its result is returned. If it fails, null is returned,
     * but the buffer's position is not advanced.
     *
     * @param parser The parser to make optional.
     * @return A Cursive parser that returns the result of the inner parser or null if it fails (without consuming input).
     */
    public static <T> Cursive<T> optional(Cursive<T> parser) {
        Objects.requireNonNull(parser, "Parser cannot be null");
        return buffer -> {
            int originalPosition = buffer.position();
            T result = parser.apply(buffer);
            if (result == null) {
                buffer.position(originalPosition); // Don't advance on failure
            }
            return result;
        };
    }

    /**
     * Creates a parser that applies another parser zero or more times.
     * It returns a list of all successful results.
     *
     * @param parser The parser to apply repeatedly.
     * @return A Cursive parser that returns a List of results (possibly empty).
     */
    public static <T> Cursive<List<T>> many(Cursive<T> parser) {
        Objects.requireNonNull(parser, "Parser cannot be null");
        return buffer -> {
            List<T> results = new ArrayList<>();
            while (true) {
                int originalPosition = buffer.position();
                T result = parser.apply(buffer);
                if (result != null) {
                    results.add(result);
                } else {
                    buffer.position(originalPosition); // Rewind to avoid partial consumption if next attempt fails
                    break;
                }
            }
            return results;
        };
    }

    /**
     * Creates a parser that applies another parser one or more times.
     * It returns a list of all successful results. Fails if the parser cannot succeed at least once.
     *
     * @param parser The parser to apply repeatedly.
     * @return A Cursive parser that returns a List of results (at least one), or null if it cannot succeed once.
     */
    public static <T> Cursive<List<T>> many1(Cursive<T> parser) {
        Objects.requireNonNull(parser, "Parser cannot be null");
        return buffer -> {
            List<T> results = many(parser).apply(buffer);
            if (results != null && !results.isEmpty()) {
                return results;
            }
            return null; // Must succeed at least once
        };
    }

    /**
     * Creates a parser that applies the given parser and then applies a terminator parser.
     * The result of the terminator is discarded.
     *
     * @param parser The main parser.
     * @param terminator The parser that follows and is discarded.
     * @param <T> The type of the main parser's result.
     * @param <U> The type of the terminator parser's result (discarded).
     * @return A Cursive parser that returns the result of the main parser.
     */
    public static <T, U> Cursive<T> terminated(Cursive<T> parser, Cursive<U> terminator) {
        Objects.requireNonNull(parser, "Parser cannot be null");
        Objects.requireNonNull(terminator, "Terminator cannot be null");
        return buffer -> {
            int originalPosition = buffer.position();
            T result = parser.apply(buffer);
            if (result != null) {
                U terminatorResult = terminator.apply(buffer);
                if (terminatorResult != null) {
                    return result;
                }
            }
            buffer.position(originalPosition); // Rewind on failure of either
            return null;
        };
    }

    /**
     * Creates a parser that applies a parser, then applies a separator parser, and repeats.
     * The result is a list of items, with separators discarded.
     *
     * @param itemParser The parser for individual items.
     * @param separatorParser The parser for the separator.
     * @param <T> The type of the item parser's result.
     * @param <S> The type of the separator parser's result (discarded).
     * @return A Cursive parser that returns a List of results (possibly empty).
     */
    public static <T, S> Cursive<List<T>> sepBy(Cursive<T> itemParser, Cursive<S> separatorParser) {
        Objects.requireNonNull(itemParser, "Item parser cannot be null");
        Objects.requireNonNull(separatorParser, "Separator parser cannot be null");
        return buffer -> {
            List<T> results = new ArrayList<>();
            T firstItem = itemParser.apply(buffer);
            if (firstItem == null) {
                return results; // No items, empty list
            }
            results.add(firstItem);

            while (true) {
                int originalPosition = buffer.position();
                S separatorResult = separatorParser.apply(buffer);
                if (separatorResult != null) {
                    T nextItem = itemParser.apply(buffer);
                    if (nextItem != null) {
                        results.add(nextItem);
                    } else {
                        buffer.position(originalPosition); // Separator matched, but no item, rewind and stop
                        break;
                    }
                } else {
                    buffer.position(originalPosition); // No separator, stop
                    break;
                }
            }
            return results;
        };
    }

    /**
     * Creates a parser that applies a parser, then applies a separator parser, and repeats.
     * Fails if there is no first item or separator cannot be found after an item.
     *
     * @param itemParser The parser for individual items.
     * @param separatorParser The parser for the separator.
     * @param <T> The type of the item parser's result.
     * @param <S> The type of the separator parser's result (discarded).
     * @return A Cursive parser that returns a List of results (at least one).
     */
    public static <T, S> Cursive<List<T>> sepBy1(Cursive<T> itemParser, Cursive<S> separatorParser) {
        Objects.requireNonNull(itemParser, "Item parser cannot be null");
        Objects.requireNonNull(separatorParser, "Separator parser cannot be null");
        return buffer -> {
            List<T> results = sepBy(itemParser, separatorParser).apply(buffer);
            if (results != null && !results.isEmpty()) {
                return results;
            }
            return null; // Must succeed at least once
        };
    }

    /**
     * Applies a parser, then transforms its result based on a predicate.
     *
     * @param parser The parser to apply.
     * @param predicate The predicate to test the result.
     * @param <T> The type of the parser's result.
     * @return A Cursive parser that returns the result if the predicate is true, null otherwise.
     */
    public static <T> Cursive<T> satisfy(Cursive<T> parser, Predicate<T> predicate) {
        Objects.requireNonNull(parser, "Parser cannot be null");
        Objects.requireNonNull(predicate, "Predicate cannot be null");
        return buffer -> {
            int originalPosition = buffer.position();
            T result = parser.apply(buffer);
            if (result != null && predicate.test(result)) {
                return result;
            }
            buffer.position(originalPosition); // Rewind if predicate fails
            return null;
        };
    }
}
