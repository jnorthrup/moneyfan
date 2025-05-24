package com.bbcursive.core;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.function.Function;

public final class BBCombinator {
    private BBCombinator() {} // Non-instantiable

    /**
     * Applies parsers in sequence. Fails if any parser in the sequence fails.
     * This version returns the result of the LAST parser in the sequence.
     * To get all results, a List-returning version or a Pair-returning version would be needed.
     */
    public static <R> Parser<R> sequence(Parser<?>... parsers) {
        if (parsers == null || parsers.length == 0) {
            // Returns a parser that consumes nothing and yields null (or a specific marker)
            return input -> ParseResult.success(null, input, input.position());
        }
        return input -> {
            ByteBuffer currentBuffer = input;
            int originalPos = input.position();
            ParseResult<?> lastResult = null;

            for (Parser<?> p : parsers) {
                lastResult = p.parse(currentBuffer);
                if (!lastResult.success()) {
                    // Backtrack the original input buffer to where this sequence started
                    input.position(originalPos);
                    return ParseResult.failure(input, "Sequence failed at " + p.getClass().getSimpleName() + ": " + lastResult.errorMessage().orElse(""));
                }
                currentBuffer = lastResult.remaining();
            }
            // Successfully parsed all, cast last result (potentially unsafe if types don't match R)
            // A safer sequence would require explicit type handling or pair/tuple results.
            return ParseResult.success((R) lastResult.value().orElse(null), currentBuffer, originalPos);
        };
    }
    
    /**
     * A sequence parser that returns a List of all successful non-Void results.
     */
    public static Parser<List<Object>> sequenceToList(Parser<?>... parsers) {
        return input -> {
            ByteBuffer currentBuffer = input;
            int originalPos = input.position();
            List<Object> results = new ArrayList<>();

            for (Parser<?> p : parsers) {
                ParseResult<?> result = p.parse(currentBuffer);
                if (!result.success()) {
                    input.position(originalPos); // Backtrack
                    return ParseResult.failure(input, "SequenceToList failed at " + p + ": " + result.errorMessage().orElse(""));
                }
                currentBuffer = result.remaining();
                result.value().ifPresent(results::add); // Add if value is present
            }
            return ParseResult.success(results, currentBuffer, originalPos);
        };
    }


    /**
     * Tries parsers in order until one succeeds. Fails if all fail.
     */
    public static <R> Parser<R> choice(Parser<? extends R>... parsers) {
        return input -> {
            int originalPos = input.position(); // To reset for each choice
            for (Parser<? extends R> p : parsers) {
                // Each choice needs to try from the original starting position of this 'choice' combinator
                input.position(originalPos);
                ParseResult<? extends R> result = p.parse(input);
                if (result.success()) {
                    return (ParseResult<R>) result; // Safe cast due to ? extends R
                }
                // If failed, input should already be reset by the failing parser p or we reset it here
                // For robustness, ensure input is reset if p didn't.
                input.position(originalPos);
            }
            return ParseResult.failure(input, "All choices failed.");
        };
    }

    /**
     * Optionally applies a parser. Always succeeds.
     * Returns an Optional of the parser's result.
     */
    public static <R> Parser<Optional<R>> optional(Parser<R> parser) {
        return input -> {
            int originalPos = input.position();
            ParseResult<R> result = parser.parse(input);
            if (result.success()) {
                return ParseResult.success(result.value(), result.remaining(), originalPos);
            } else {
                // Parser failed, but it's optional. Reset position and succeed with empty Optional.
                input.position(originalPos);
                return ParseResult.success(Optional.empty(), input, originalPos);
            }
        };
    }

    /**
     * Applies a parser zero or more times. Always succeeds.
     * Collects successful results into a List.
     */
    public static <R> Parser<List<R>> many0(Parser<R> parser) {
        return input -> {
            List<R> results = new ArrayList<>();
            ByteBuffer currentBuffer = input;
            int originalPos = input.position(); // Position at the start of many0

            while (true) {
                int posBeforeItem = currentBuffer.position();
                ParseResult<R> result = parser.parse(currentBuffer);
                if (result.success()) {
                    result.value().ifPresent(results::add); // Add value if present
                    currentBuffer = result.remaining();
                    // Check for progress; if parser succeeded but consumed nothing, break to prevent infinite loop
                    if (currentBuffer.position() == posBeforeItem && result.value().isPresent()) {
                         // This indicates a parser that can succeed without consuming input, e.g. opt(p) or a parser for empty string.
                         // If it produced a value, we might allow it (e.g. list of optionals).
                         // For now, let's be strict: if it consumes nothing and gives a value, it's suspicious for 'many'.
                         // This depends on the exact semantics desired for parsers that succeed on empty input.
                         // A common strategy: if a 'many' item parse consumes no input, stop.
                    } else if (currentBuffer.position() == posBeforeItem) {
                         break; // No progress, stop.
                    }
                } else {
                    // Parser failed, so we're done with the "many" part.
                    // Reset to position before this failed attempt.
                    currentBuffer.position(posBeforeItem);
                    break;
                }
            }
            return ParseResult.success(results, currentBuffer, originalPos);
        };
    }

    /**
     * Applies a parser one or more times. Fails if the first attempt fails.
     * Collects successful results into a List.
     */
    public static <R> Parser<List<R>> many1(Parser<R> parser) {
        return input -> {
            int originalPos = input.position();
            ParseResult<List<R>> firstResult = many0(parser).parse(input); // Try many0
            if (firstResult.success() && !firstResult.value().orElseThrow().isEmpty()) {
                return firstResult;
            }
            // If many0 succeeded but list is empty, or if many0 somehow failed (shouldn't)
            input.position(originalPos); // Ensure reset
            return ParseResult.failure(input, "Expected at least one match for many1.");
        };
    }

    // --- Higher-Order Parser Transformations ---
    /**
     * Transforms the successful result of a parser.
     */
    public static <A, B> Parser<B> map(Parser<A> parser, Function<A, B> fn) {
        return input -> {
            ParseResult<A> resultA = parser.parse(input);
            if (resultA.success()) {
                A valueA = resultA.value().orElse(null); // Handle optional if A can be Void
                if (valueA == null && !resultA.value().isPresent()){ // case for Parser<Void>
                     try {
                        // If fn can accept null (e.g. Function<Void, B>), allow it.
                        // This is a bit tricky with generics for Void.
                        // Often, mapping a Void parser is used to inject a constant.
                        return ParseResult.success(fn.apply(null), resultA.remaining(), resultA.originalPosition());
                    } catch (NullPointerException npe){
                        // if fn strictly needs non-null A and A was Void/absent
                        return ParseResult.failure(resultA.remaining(), "Map function requires non-null input, but parser yielded no value.");
                    }
                }
                return ParseResult.success(fn.apply(valueA), resultA.remaining(), resultA.originalPosition());
            }
            return ParseResult.failure(resultA.remaining(), resultA.errorMessage().orElse("Map source parser failed."));
        };
    }
}
