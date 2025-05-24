package com.example.bbcursive.util;

import com.example.bbcursive.BBAtom;
import com.example.bbcursive.BBCombinator;
import com.example.bbcursive.Cursive;

import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.function.Function;

/**
 * Utility class providing common, pre-composed byte parsers using `bbcursive`.
 */
public class ByteParsers {

    /**
     * Parses a byte array of a specific length.
     * @param length The number of bytes to read.
     * @return A Cursive parser for a byte array.
     */
    public static Cursive<byte[]> byteArray(int length) {
        return buffer -> {
            if (buffer.remaining() < length) return null;
            byte[] bytes = new byte[length];
            buffer.get(bytes);
            return bytes;
        };
    }

    /**
     * Parses a short.
     * @return A Cursive parser for a short.
     */
    public static Cursive<Short> shortParser() {
        return BBAtom.shortP();
    }

    /**
     * Parses an integer.
     * @return A Cursive parser for an integer.
     */
    public static Cursive<Integer> intParser() {
        return BBAtom.intP();
    }

    /**
     * Parses a long.
     * @return A Cursive parser for a long.
     */
    public static Cursive<Long> longParser() {
        return BBAtom.longP();
    }

    /**
     * Parses a double.
     * @return A Cursive parser for a double.
     */
    public static Cursive<Double> doubleParser() {
        return BBAtom.doubleP();
    }

    /**
     * Parses a fixed-length string (UTF-8).
     * @param length The number of bytes to read for the string.
     * @return A Cursive parser for a String.
     */
    public static Cursive<String> stringParser(int length) {
        return BBAtom.string(length);
    }

    /**
     * Parses a sequence of integers.
     * @param count The number of integers to read.
     * @return A Cursive parser for a List of Integers.
     */
    public static Cursive<List<Integer>> intArray(int count) {
        // Create a list of 'count' integer parsers and sequence them.
        Cursive<Integer>[] parsers = new Cursive[count];
        for (int i = 0; i < count; i++) {
            parsers[i] = BBAtom.intP();
        }
        return BBCombinator.sequence(parsers);
    }

    /**
     * Parses a C-style null-terminated string. Reads bytes until a null byte (0x00) is encountered.
     * The null byte is consumed but not included in the resulting string.
     *
     * @param maxLength The maximum number of bytes to read before assuming no null terminator.
     *                  This is to prevent infinite loops on malformed data.
     * @return A Cursive parser that returns a String. Returns null if max length is reached without terminator.
     */
    public static Cursive<String> nullTerminatedString(int maxLength) {
        return buffer -> {
            int originalPos = buffer.position();
            StringBuilder sb = new StringBuilder();
            int bytesRead = 0;

            while (buffer.hasRemaining() && bytesRead < maxLength) {
                byte b = buffer.get();
                bytesRead++;
                if (b == 0x00) {
                    return sb.toString();
                }
                sb.append((char) b); // Assuming single-byte characters for simplicity, or
                                     // collect bytes and then decode. For UTF-8, need more sophisticated logic.
            }
            buffer.position(originalPos); // Rewind if no null terminator within max length
            return null;
        };
    }

    /**
     * Parses a line of text terminated by newline ('\n') or carriage return-newline ("\r\n").
     * The terminator is consumed but not included in the result.
     *
     * @param maxLength The maximum length of the line to read.
     * @return A Cursive parser that returns a String. Returns null if no line terminator within max length.
     */
    public static Cursive<String> line() {
        return buffer -> {
            int originalPos = buffer.position();
            StringBuilder sb = new StringBuilder();

            while (buffer.hasRemaining()) {
                byte b = buffer.get();
                if (b == '\n') {
                    // Check for preceding '\r'
                    if (sb.length() > 0 && sb.charAt(sb.length() - 1) == '\r') {
                        sb.deleteCharAt(sb.length() - 1);
                    }
                    return sb.toString();
                } else if (b == '\r') {
                    // Peek ahead for '\n', if not present, treat as end of line (or part of content if not followed by \n)
                    if (buffer.hasRemaining() && buffer.get(buffer.position()) == '\n') {
                        buffer.get(); // Consume '\n'
                        return sb.toString();
                    }
                    // If just '\r' and no '\n' follows, append it or treat as end of line
                    // For simplicity, let's treat it as end of line if no \n follows immediately.
                    // Or, if not followed by \n, consider it part of the content.
                    // This is an ambiguous case in some CSV/text formats. Standard is \r\n or \n.
                    sb.append((char)b); // Append for now
                } else {
                    sb.append((char)b);
                }
            }
            // If buffer exhausted without newline, return remaining as a line
            // Or, for strict parsing, return null if no proper terminator found.
            if (sb.length() > 0) {
                return sb.toString();
            }
            buffer.position(originalPos); // No content, no line
            return null;
        };
    }
}
