package com.moneyfan.dsel.core;

import java.util.function.Supplier;

/**
 * Conceptual type alias for Series&lt;Join&lt;Object, Supplier&lt;ColumnMeta&gt;&gt;&gt;, which translates to:
 * Join&lt;Integer, Function&lt;Integer, Join&lt;Object, Supplier&lt;Join&lt;String, TypeMemento&gt;&gt;&gt;&gt;&gt;.
 * <p>
 * Represents a row vector, a series of cells where each cell is a Join of its value and a supplier for its metadata.
 * Instances are created using {@link Types#rv(int, java.util.function.Function)}.
 */
public interface RowVec {}
