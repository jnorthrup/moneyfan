package com.yourdomain.bbcursive.core;

import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.nio.ByteBuffer;
import java.util.Objects;
import java.util.Optional;

/**
 * Represents the result of a parsing operation, encapsulating the parsed value,
 * the state of the ByteBuffer after parsing, and whether the parsing was successful.
 *
 * @param <T> The type of the parsed value.
 */
public final class ParseResult<T> {
    private final @Nullable T value;
    private final @Nullable ByteBuffer remainingBuffer; // The buffer *after* parsing
    private final boolean success;

    private ParseResult(@Nullable T value, @Nullable ByteBuffer remainingBuffer, boolean success) {
        this.value = value;
        this.remainingBuffer = remainingBuffer;
        this.success = success;
    }

    /**
     * Creates a successful ParseResult.
     * @param value The parsed value.
     * @param remainingBuffer The ByteBuffer state after successful parsing.
     * @param <T> The type of the parsed value.
     * @return A successful ParseResult.
     */
    public static <T> @NotNull ParseResult<T> success(@Nullable T value, @NotNull ByteBuffer remainingBuffer) {
        return new ParseResult<>(value, remainingBuffer, true);
    }

    /**
     * Creates a failed ParseResult.
     * @param <T> The type of the parsed value.
     * @return A failed ParseResult.
     */
    public static <T> @NotNull ParseResult<T> failure() {
        return new ParseResult<>(null, null, false);
    }

    public boolean isSuccess() {
        return success;
    }

    public boolean isFailure() {
        return !success;
    }

    /**
     * Returns the parsed value if successful, otherwise throws IllegalStateException.
     * @return The parsed value.
     * @throws IllegalStateException if the parsing was not successful.
     */
    public @Nullable T getValue() {
        if (!success) {
            throw new IllegalStateException("Cannot get value from a failed ParseResult.");
        }
        return value;
    }

    /**
     * Returns an Optional containing the parsed value if successful, otherwise an empty Optional.
     * @return An Optional of the parsed value.
     */
    public @NotNull Optional<T> getOptionalValue() {
        return Optional.ofNullable(value);
    }

    /**
     * Returns the ByteBuffer state after parsing if successful, otherwise throws IllegalStateException.
     * @return The remaining ByteBuffer.
     * @throws IllegalStateException if the parsing was not successful.
     */
    public @NotNull ByteBuffer getRemainingBuffer() {
        if (!success || remainingBuffer == null) {
            throw new IllegalStateException("Cannot get remaining buffer from a failed ParseResult.");
        }
        return remainingBuffer;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        ParseResult<?> that = (ParseResult<?>) o;
        return success == that.success && Objects.equals(value, that.value) && Objects.equals(remainingBuffer, that.remainingBuffer);
    }

    @Override
    public int hashCode() {
        return Objects.hash(value, remainingBuffer, success);
    }

    @Override
    public String toString() {
        return "ParseResult{" +
               "value=" + value +
               ", remainingBuffer.position=" + (remainingBuffer != null ? remainingBuffer.position() : "null") +
               ", remainingBuffer.limit=" + (remainingBuffer != null ? remainingBuffer.limit() : "null") +
               ", success=" + success +
               '}';
    }
}
