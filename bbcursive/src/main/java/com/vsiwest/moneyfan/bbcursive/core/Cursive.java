package com.vsiwest.moneyfan.bbcursive.core;

import java.nio.ByteBuffer;
import java.util.function.Function;

@FunctionalInterface
public interface Cursive<T> extends Function<ByteBuffer, T> {

    @Override
    T apply(ByteBuffer buffer);

}
