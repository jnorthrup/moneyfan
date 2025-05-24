package com.vsiwest.moneyfan.bbcursive.core;

import java.nio.ByteBuffer;
import java.util.Optional;

public record ParseResult<R>(
    Optional<R> value,       // The parsed value, if successful
    ByteBuffer remaining,    // The ByteBuffer positioned after the parsed segment (or original on failure)
    int originalPosition,    // Original position before this parse attempt (for backtracking/error reporting)
    boolean success,
    Optional<String> errorMessage // Optional error message on failure
) {

    public R orElseThrow() {
        return value.orElseThrow(() -> new IllegalStateException(errorMessage.orElse("Parse failure")));
    }
}
