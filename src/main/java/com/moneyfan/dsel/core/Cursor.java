package com.moneyfan.dsel.core;

/**
 * Conceptual type alias for Series&lt;RowVec&gt;, which translates to:
 * Join&lt;Integer, Function&lt;Integer, RowVec&gt;&gt; or
 * Join&lt;Integer, Function&lt;Integer, Join&lt;Integer, Function&lt;Integer, Join&lt;Object, Supplier&lt;Join&lt;String, TypeMemento&gt;&gt;&gt;&gt;&gt;&gt;&gt;.
 * <p>
 * Represents a cursor over rows, where each row is a RowVec.
 * Instances are created using {@link Types#cr(int, java.util.function.Function)}.
 */
public interface Cursor {}
