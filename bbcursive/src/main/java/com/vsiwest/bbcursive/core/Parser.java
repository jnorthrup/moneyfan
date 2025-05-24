package com.vsiwest.bbcursive.core;

import org.jetbrains.annotations.NotNull;
import java.nio.ByteBuffer;

@FunctionalInterface
public interface Parser<R> {
    @NotNull
    ParseResult<R> parse(@NotNull ByteBuffer input);
}
