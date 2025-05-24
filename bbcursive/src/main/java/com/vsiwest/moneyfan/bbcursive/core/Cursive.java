package com.vsiwest.moneyfan.bbcursive.core;

import java.nio.ByteBuffer;
import java.util.function.Function;

/**
 * Represents a functional parser for ByteBuffer.
 * A parser takes a ByteBuffer and attempts to produce a result of type T.
 * If parsing is successful, it returns the result; otherwise, it can throw an exception
 * or return an Optional/Either type for more robust error handling.
 * For this design, we'll keep it simple as a Function for successful parsing.
 * Errors will be handled by returning null or throwing.
 *
 * @param <T> The type of the result produced by the parser.
 */
@FunctionalInterface
public interface Cursive<T> extends Function<ByteBuffer, T> {

    /**
     * Applies the parsing logic to the given ByteBuffer.
     * The ByteBuffer's position will be advanced if parsing is successful.
     *
     * @param buffer The ByteBuffer to parse.
     * @return The parsed result of type T.
     * @throws java.util.NoSuchElementException if parsing fails (e.g., end of buffer reached unexpectedly).
     * @throws IllegalArgumentException if the buffer content does not match the expected format.
     */
    @Override
    T apply(ByteBuffer buffer);

    /**
     * Composes this parser with another parser. The output of this parser
     * (which must be a ByteBuffer representing the remaining input) is then
     * fed as input to the next parser.
     *
     * @param after The function to apply after this parser.
     * @param <V> The type of the result of the `after` function.
     * @return A new Cursive parser that applies this parser, then the `after` function.
     */
    default <V> Cursive<V> andThen(Function<? super ByteBuffer, ? extends V> after) {
        return buffer -> {
            T result = this.apply(buffer);
            // Assuming 'result' is the remaining buffer or a specific marker for composition
            // This method might need adjustment based on how parsers consume/return buffers.
            // For now, let's assume 'apply' implicitly advances the buffer's position.
            // So, 'after' simply continues parsing from the current buffer state.
            if (result != null) { // Simple success check
                return after.apply(buffer); // Continue with the same buffer
            }
            return null; // Or throw, depending on error strategy
        };
    }

    /**
     * Maps the result of this parser to a new type using a given function.
     *
     * @param mapper The function to transform the result.
     * @param <R> The type of the new parsed value.
     * @return A new Cursive parser that applies this parser and then transforms its result.
     */
    default <R> Cursive<R> map(Function<? super T, ? extends R> mapper) {
        return buffer -> {
            T result = this.apply(buffer);
            return (result != null) ? mapper.apply(result) : null;
        };
    }

    /**
     * Chains two parsers: if the first parser succeeds, its result is used
     * to create the second parser, which then parses the remaining input.
     * This is useful for context-dependent parsing.
     *
     * @param nextParserFactory A function that takes the result of the first parser
     *                          and returns a new parser for the remaining input.
     * @param <R> The type of the result of the chained parser.
     * @return A new Cursive parser that applies the chaining logic.
     */
    default <R> Cursive<R> flatMap(Function<? super T, ? extends Cursive<R>> nextParserFactory) {
        return buffer -> {
            T intermediateResult = this.apply(buffer);
            if (intermediateResult != null) {
                Cursive<R> nextParser = nextParserFactory.apply(intermediateResult);
                if (nextParser != null) {
                    return nextParser.apply(buffer);
                }
            }
            return null;
        };
    }
}
