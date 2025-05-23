package com.moneyfan.common;

import java.util.Iterator;
import java.util.function.Function;
import java.util.function.Predicate;

/**
 * A DSEL-focused cursor, similar to an Iterator but with chainable operations for syntactic sugar.
 * This design draws inspiration from jQuery's chaining, allowing fluent method calls to build pipelines.
 * Operations are unary and return new cursors for composability, insulating users from low-level details.
 */
public interface Cursor extends Series<RowVec> {

 }
