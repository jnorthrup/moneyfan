package com.bbcursive.core;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;

public final class BBAtom {
    private BBAtom() {} // Non-instantiable

    // --- Byte Level Atoms ---
    public static Parser<Byte> byteIs(byte expectedByte) {
        return input -> {
            int originalPos = input.position();
            if (input.hasRemaining()) {
                byte actualByte = input.get();
                if (actualByte == expectedByte) {
                    return ParseResult.success(actualByte, input, originalPos);
                }
                input.position(originalPos); // backtrack
            }
            return ParseResult.failure(input, "Expected byte " + expectedByte + " but found end or different.");
        };
    }

    public static Parser<byte[]> bytesMatch(byte[] expectedBytes) {
        final byte[] expected = Arrays.copyOf(expectedBytes, expectedBytes.length); // Defensive copy
        return input -> {
            int originalPos = input.position();
            if (input.remaining() >= expected.length) {
                byte[] actual = new byte[expected.length];
                input.get(actual);
                if (Arrays.equals(actual, expected)) {
                    return ParseResult.success(actual, input, originalPos);
                }
                input.position(originalPos); // backtrack
            }
            return ParseResult.failure(input, "Expected bytes " + Arrays.toString(expected) + " not found.");
        };
    }

    // --- Character Level Atoms (Charset sensitive) ---
    public static Parser<Character> charIs(char expectedChar, Charset charset) {
        // This is tricky for variable-length charsets like UTF-8.
        // For simplicity, often done by parsing a string of length 1.
        // A more robust char parser would handle multi-byte chars.
        // Here's a simplified version (might be better to use stringIs for single char):
        return BBCombinator.map(stringIs(String.valueOf(expectedChar), charset), s -> s.charAt(0));
    }

    public static Parser<String> stringIs(String expectedString, Charset charset) {
        final byte[] expectedBytes = expectedString.getBytes(charset);
        return input -> {
            ParseResult<byte[]> byteResult = bytesMatch(expectedBytes).parse(input);
            if (byteResult.success()) {
                // The string is already known, no need to re-decode if this is the only purpose
                return ParseResult.success(expectedString, byteResult.remaining(), byteResult.originalPosition());
            }
            return ParseResult.failure(input, "Expected string '" + expectedString + "' not found.");
        };
    }

    // --- Primitive Type Parsers (ByteOrder sensitive) ---
    public static Parser<Short> parseShort(ByteOrder order) {
        return input -> {
            int originalPos = input.position();
            if (input.remaining() >= Short.BYTES) {
                ByteOrder originalOrder = input.order();
                input.order(order);
                short val = input.getShort();
                input.order(originalOrder); // Restore original order
                return ParseResult.success(val, input, originalPos);
            }
            return ParseResult.failure(input, "Not enough bytes for a Short.");
        };
    }

    public static Parser<Integer> parseInt(ByteOrder order) {
        return input -> {
            int originalPos = input.position();
            if (input.remaining() >= Integer.BYTES) {
                ByteOrder originalOrder = input.order();
                input.order(order);
                int val = input.getInt();
                input.order(originalOrder);
                return ParseResult.success(val, input, originalPos);
            }
            return ParseResult.failure(input, "Not enough bytes for an Integer.");
        };
    }

    public static Parser<Long> parseLong(ByteOrder order) {
         return input -> {
            int originalPos = input.position();
            if (input.remaining() >= Long.BYTES) {
                ByteOrder originalOrder = input.order();
                input.order(order);
                long val = input.getLong();
                input.order(originalOrder);
                return ParseResult.success(val, input, originalPos);
            }
            return ParseResult.failure(input, "Not enough bytes for a Long.");
        };
    }
    // Add parseFloat, parseDouble similarly...

    // --- Fixed Length Data Parsers ---
    public static Parser<String> parseFixedString(int length, Charset charset) {
        return input -> {
            int originalPos = input.position();
            if (input.remaining() >= length) {
                byte[] arr = new byte[length];
                input.get(arr);
                // ISAM specific: often null-terminated within the fixed length.
                // Find first null byte to determine actual string length.
                int actualLen = 0;
                while(actualLen < length && arr[actualLen] != 0) {
                    actualLen++;
                }
                String val = new String(arr, 0, actualLen, charset);
                return ParseResult.success(val, input, originalPos);
            }
            return ParseResult.failure(input, "Not enough bytes for fixed string of length " + length);
        };
    }

    public static Parser<byte[]> parseFixedBlob(int length) {
        return input -> {
            int originalPos = input.position();
            if (input.remaining() >= length) {
                byte[] arr = new byte[length];
                input.get(arr);
                return ParseResult.success(arr, input, originalPos);
            }
            return ParseResult.failure(input, "Not enough bytes for fixed blob of length " + length);
        };
    }

    // --- Utility for "consuming" a fixed number of bytes without a specific value ---
    public static Parser<Void> skip(int numBytes) {
        return input -> {
            int originalPos = input.position();
            if (input.remaining() >= numBytes) {
                input.position(originalPos + numBytes);
                return ParseResult.successVoid(input, originalPos);
            }
            return ParseResult.failure(input, "Not enough bytes to skip " + numBytes);
        };
    }

    // --- Zero-Copy View Operations as Parsers (these don't consume, but transform the input view for subsequent parsers) ---
    // These are more like pre-processors for other parsers or for debugging.
    // A true "view" parser would likely be a combinator that takes another parser.

    /**
     * A parser that always succeeds and returns a slice of the input buffer
     * from its current position to its limit. The original buffer's position is advanced
     * to its limit.
     */
    public static Parser<ByteBuffer> takeSlice() {
        return input -> {
            int originalPos = input.position();
            ByteBuffer slice = input.slice(); // Creates a view from pos to limit
            input.position(input.limit());   // Advance original buffer
            return ParseResult.success(slice, input, originalPos);
        };
    }

    /**
     * Takes a slice of a specific length.
     */
    public static Parser<ByteBuffer> takeSlice(int length) {
        return input -> {
            int originalPos = input.position();
            if (input.remaining() >= length) {
                ByteBuffer view = input.slice();
                view.limit(length);
                input.position(originalPos + length); // Advance original buffer
                return ParseResult.success(view, input, originalPos);
            }
            return ParseResult.failure(input, "Not enough bytes to take slice of length " + length);
        };
    }
}
