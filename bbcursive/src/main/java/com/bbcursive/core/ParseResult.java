package com.bbcursive.core;

import java.nio.ByteBuffer;
import java.util.Objects;
import java.util.Optional; // Using Optional for the value

/**
 * Represents the result of a parsing operation.
 *
 * @param <R> The type of the parsed value.
 */
public record ParseResult<R>(
    Optional<R> value,       // The parsed value, if successful
    ByteBuffer remaining,    // The ByteBuffer positioned after the parsed segment (or original on failure)
    int originalPosition,    // Original position before this parse attempt (for backtracking/error reporting)
    boolean success,
    Optional<String> errorMessage // Optional error message on failure
) {

    /**
     * Creates a success result.
     */
    public static <R> ParseResult<R> success(R value, ByteBuffer remainingBuffer, int originalPosition) {
        Objects.requireNonNull(value, "Value cannot be null for a successful parse.");
        Objects.requireNonNull(remainingBuffer, "Remaining buffer cannot be null.");
        return new ParseResult<>(Optional.of(value), remainingBuffer, originalPosition, true, Optional.empty());
    }

    /**
     * Creates a success result for parsers that consume input but don't produce a specific value (e.g., matching a literal).
     * The value will be a placeholder or could be an empty Optional if R is Void.
     */
    public static ParseResult<Void> successVoid(ByteBuffer remainingBuffer, int originalPosition) {
        Objects.requireNonNull(remainingBuffer, "Remaining buffer cannot be null.");
        return new ParseResult<>(Optional.empty(), remainingBuffer, originalPosition, true, Optional.empty());
    }

    /**
     * Creates a failure result. The remaining buffer is typically the original buffer at its original position.
     */
    public static <R> ParseResult<R> failure(ByteBuffer originalBufferAtStart, String message) {
        Objects.requireNonNull(originalBufferAtStart, "Original buffer cannot be null for failure.");
        return new ParseResult<>(Optional.empty(), originalBufferAtStart, originalBufferAtStart.position(), false, Optional.ofNullable(message));
    }

    public R orElseThrow() {
        return value.orElseThrow(() -> new IllegalStateException(errorMessage.orElse("Parse failure with no value.")));
    }
}
