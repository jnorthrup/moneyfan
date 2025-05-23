package com.moneyfan.dsel.core;

import java.util.function.Function;

/**
 * Conceptual type alias for Join&lt;Integer, Function&lt;Integer, T&gt;&gt;.
 * Represents a sequence of elements of type T, defined by a size and a generator function.
 * Instances are created using {@link Types#sr(int, Function)}.
 */
public interface Series<T> {
}
