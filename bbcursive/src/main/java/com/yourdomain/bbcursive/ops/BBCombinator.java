package com.yourdomain.bbcursive.ops;

import com.yourdomain.bbcursive.core.Cursive;
import com.yourdomain.bbcursive.core.ParseResult;
import org.jetbrains.annotations.NotNull;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.function.Predicate;

/**
 * {@code BBCombinator} provides higher-order functions to combine simple {@link Cursive} parsers
 * into more complex parsing grammars. These functions encapsulate common parsing patterns
 * like sequencing, choice, repetition, and optionality.
 *
 * This enum acts as a utility class, leveraging the enum's static-like nature
 * to group utility methods.
 */
public enum BBCombinator {
    ; // No instances

    /**
     * Creates a parser that succeeds if all provided parsers succeed in sequence.
     * The result is a list of the parsed values.
     *
     * @param parsers An array of Cursive parsers.
     * @param <T> The common supertype of values produced by the parsers.
     * @return A Cursive parser that produces a List of parsed values.
     */
    @SafeVarargs
    public static <T> @NotNull Cursive<List<T>> sequence(@NotNull Cursive<? extends T>... parsers) {
        return buffer -> {
            ByteBuffer currentBuffer = buffer.duplicate(); // Work on a duplicate to allow rollback
            List<T> results = new ArrayList<>();
            for (Cursive<? extends T> parser : parsers) {
                ParseResult<? extends T> result = parser.parse(currentBuffer);
                if (result.isSuccess()) {
                    results.add(result.getValue());
                    currentBuffer = result.getRemainingBuffer(); // Update buffer for next parser in sequence
                } else {
                    return ParseResult.failure(); // Any failure in sequence means overall failure
                }
            }
            // If all succeeded, update the original buffer's position and return success
            buffer.position(currentBuffer.position());
            return ParseResult.success(results, buffer);
        };
    }

    /**
     * Creates a parser that succeeds if any of the provided parsers succeed.
     * It tries parsers in order and returns the first successful result.
     *
     * @param parsers An array of Cursive parsers.
     * @param <T> The common supertype of values produced by the parsers.
     * @return A Cursive parser that produces a single parsed value.
     */
    @SafeVarargs
    public static <T> @NotNull Cursive<T> choice(@NotNull Cursive<? extends T>... parsers) {
        return buffer -> {
            ByteBuffer originalBuffer = buffer.duplicate(); // Preserve original state for each attempt
            for (Cursive<? extends T> parser : parsers) {
                originalBuffer.position(buffer.position()); // Reset position for this attempt
                ParseResult<? extends T> result = parser.parse(originalBuffer);
                if (result.isSuccess()) {
                    buffer.position(originalBuffer.position()); // Update original buffer on success
                    return ParseResult.success(result.getValue(), buffer);
                }
            }
            return ParseResult.failure(); // All choices failed
        };
    }

    /**
     * Creates a parser that tries to apply the given parser, but succeeds even if it fails.
     * If the parser succeeds, its result is returned wrapped in an Optional.
     * If it fails, an empty Optional is returned, and the buffer position is not changed.
     *
     * @param parser The Cursive parser to make optional.
     * @param <T> The type of the value produced by the parser.
     * @return A Cursive parser that produces an Optional of the parsed value.
     */
    public static <T> @NotNull Cursive<Optional<T>> optional(@NotNull Cursive<T> parser) {
        return buffer -> {
            ByteBuffer originalBuffer = buffer.duplicate(); // Preserve original state
            ParseResult<T> result = parser.parse(originalBuffer);
            if (result.isSuccess()) {
                buffer.position(originalBuffer.position()); // Update original buffer
                return ParseResult.success(Optional.ofNullable(result.getValue()), buffer);
            } else {
                return ParseResult.success(Optional.empty(), buffer); // Do not advance buffer on optional failure
            }
        };
    }

    /**
     * Creates a parser that applies the given parser zero or more times.
     * It collects all successful results into a List. Parsing stops when the parser fails.
     *
     * @param parser The Cursive parser to apply repeatedly.
     * @param <T> The type of the value produced by the parser.
     * @return A Cursive parser that produces a List of parsed values.
     */
    public static <T> @NotNull Cursive<List<T>> many(@NotNull Cursive<T> parser) {
        return buffer -> {
            ByteBuffer currentBuffer = buffer.duplicate(); // Work on a duplicate
            List<T> results = new ArrayList<>();
            while (true) {
                ByteBuffer attemptBuffer = currentBuffer.duplicate(); // For local attempt rollback
                ParseResult<T> result = parser.parse(attemptBuffer);
                if (result.isSuccess()) {
                    results.add(result.getValue());
                    currentBuffer = result.getRemainingBuffer(); // Advance current buffer
                } else {
                    break; // Parser failed, stop repetition
                }
            }
            buffer.position(currentBuffer.position()); // Update original buffer
            return ParseResult.success(results, buffer);
        };
    }

    /**
     * Creates a parser that applies the given parser one or more times.
     * Fails if the parser does not succeed at least once.
     *
     * @param parser The Cursive parser to apply repeatedly.
     * @param <T> The type of the value produced by the parser.
     * @return A Cursive parser that produces a List of parsed values.
     */
    public static <T> @NotNull Cursive<List<T>> many1(@NotNull Cursive<T> parser) {
        return buffer -> {
            ParseResult<List<T>> result = many(parser).parse(buffer);
            if (result.isSuccess() && !result.getValue().isEmpty()) {
                return result;
            }
            return ParseResult.failure();
        };
    }

    /**
     * Applies a predicate to the next byte without consuming it.
     * @param predicate The predicate to test the next byte with.
     * @return A Cursive parser that succeeds if the predicate is true for the next byte.
     */
    public static @NotNull Cursive<Byte> peekByteIf(@NotNull Predicate<Byte> predicate) {
        return buffer -> {
            if (buffer.hasRemaining()) {
                byte b = buffer.get(buffer.position()); // Peek: don't advance position
                if (predicate.test(b)) {
                    return ParseResult.success(b, buffer); // Succeed, but buffer position is unchanged
                }
            }
            return ParseResult.failure();
        };
    }
}
