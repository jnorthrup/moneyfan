package com.yourdomain.bbcursive;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.Objects;

public enum BBCombinator {
    ; // No specific instances needed

    @SafeVarargs
    public static <T> Cursive<List<T>> sequence(Cursive<? extends T>... parsers) {
        return buffer -> {
            Objects.requireNonNull(buffer, "ByteBuffer must not be null");
            int originalPosition = buffer.position();
            List<T> results = new ArrayList<>();
            try {
                for (Cursive<? extends T> parser : parsers) {
                    results.add(parser.apply(buffer));
                }
                return results;
            } catch (Exception e) {
                buffer.position(originalPosition); // Rewind on failure
                throw new IllegalStateException("Sequence parsing failed", e);
            }
        };
    }

    @SafeVarargs
    public static <T> Cursive<Optional<T>> choice(Cursive<? extends T>... parsers) {
        return buffer -> {
            Objects.requireNonNull(buffer, "ByteBuffer must not be null");
            int originalPosition = buffer.position();
            for (Cursive<? extends T> parser : parsers) {
                try {
                    return Optional.of(parser.apply(buffer));
                } catch (Exception e) {
                    buffer.position(originalPosition); // Rewind if parser fails
                }
            }
            return Optional.empty();
        };
    }

    public static <T> Cursive<Optional<T>> optional(Cursive<T> parser) {
        return buffer -> {
            Objects.requireNonNull(buffer, "ByteBuffer must not be null");
            Objects.requireNonNull(parser, "Parser must not be null");
            int originalPosition = buffer.position();
            try {
                return Optional.of(parser.apply(buffer));
            } catch (Exception e) {
                buffer.position(originalPosition); // Rewind on failure
                return Optional.empty();
            }
        };
    }

    public static <T> Cursive<List<T>> many(Cursive<T> parser) {
        return buffer -> {
            Objects.requireNonNull(buffer, "ByteBuffer must not be null");
            Objects.requireNonNull(parser, "Parser must not be null");
            List<T> results = new ArrayList<>();
            while (true) {
                int originalPosition = buffer.position();
                try {
                    results.add(parser.apply(buffer));
                } catch (Exception e) {
                    buffer.position(originalPosition); // Rewind on failure to not consume the failed byte
                    break; // Stop parsing if parser fails
                }
            }
            return results;
        };
    }
}
