package com.yourdomain.bbcursive.core;

import org.jetbrains.annotations.NotNull;

import java.nio.ByteBuffer;
import java.util.function.Function;

/**
 * The foundational functional interface for a ByteBuffer parsing operation.
 * It transforms a ByteBuffer, typically by consuming some bytes and returning the remaining buffer,
 * or by producing a parsed value and the remaining buffer state.
 *
 * This version acts as a parser combinator where 'apply' attempts to parse a value from the buffer,
 * returning a ParseResult indicating success/failure and the parsed value/remaining buffer.
 */
@FunctionalInterface
public interface Cursive<T> {

    /**
     * Attempts to parse a value of type T from the given ByteBuffer.
     * The ByteBuffer's position is expected to be updated if parsing is successful.
     *
     * @param buffer The ByteBuffer to parse from. Its position will be advanced by the parser.
     * @return A ParseResult containing either the parsed value and the buffer's state AFTER parsing,
     *         or an empty/failed result if parsing was not successful.
     */
    @NotNull
    ParseResult<T> parse(@NotNull ByteBuffer buffer);

    /**
     * Returns a composed Cursive that first applies this parser, and then applies the {@code after} parser
     * to the remaining buffer if this parser was successful. The result is a combined ParseResult.
     *
     * @param after The parser to apply after this one.
     * @param <V> The type of the value produced by the {@code after} parser.
     * @return A composed Cursive.
     */
    default <V> Cursive<V> andThen(@NotNull Cursive<V> after) {
        return buffer -> {
            ParseResult<T> firstResult = Cursive.this.parse(buffer);
            if (firstResult.isSuccess()) {
                // The 'firstResult' contains the state of the buffer after the first parse.
                // We pass the *original* buffer (whose position was updated by firstResult.getRemainingBuffer())
                // to the 'after' parser.
                return after.parse(firstResult.getRemainingBuffer()); // or just 'buffer'
            } else {
                return ParseResult.failure();
            }
        };
    }

    /**
     * Transforms the successful result of this Cursive using the given function.
     * If this parser fails, the new parser also fails.
     *
     * @param mapper The function to apply to the parsed value.
     * @param <R> The type of the new parsed value.
     * @return A Cursive that applies this parser and then transforms its result.
     */
    default <R> Cursive<R> map(@NotNull Function<T, R> mapper) {
        return buffer -> {
            ParseResult<T> result = Cursive.this.parse(buffer);
            return result.isSuccess() ? ParseResult.success(mapper.apply(result.getValue()), result.getRemainingBuffer()) : ParseResult.failure();
        };
    }
}
