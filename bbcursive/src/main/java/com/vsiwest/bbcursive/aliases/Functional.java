package com.vsiwest.bbcursive.aliases;

import com.vsiwest.bbcursive.core.Join;
import java.util.function.Function;

public interface UnaryAsyncReaction {}
public interface AsyncReaction extends Join<Integer, UnaryAsyncReaction> {}
public interface LongSeries<T> extends Join<Long, Function<Long, T>> {}
