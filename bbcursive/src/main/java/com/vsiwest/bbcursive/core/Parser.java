package com.bbcursive.core;

import java.nio.ByteBuffer;

/**
 * A functional interface representing a parser that takes a ByteBuffer
 * and attempts to parse a value of type R from it.
 *
 * @param <R> The type of the value this parser produces.
 */
@FunctionalInterface
public interface Parser<R> {
    /**
     * Applies this parser to the given ByteBuffer.
     *
     * @param input The ByteBuffer to parse from. Its state (position, limit)
     *              should be respected and updated by the parser.
     * @return A ParseResult indicating success or failure, containing the parsed
     *         value (if any) and the state of the ByteBuffer after parsing.
     */
    ParseResult<R> parse(ByteBuffer input);

    // Default methods for common combinators can be added here later
    // e.g., map, flatMap, or, andThen, optional, many, etc.
}
