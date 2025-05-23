package com.moneyfan.dsel;

/**
 * Specifies the type of join to be performed between two Series.
 */
public enum JoinType {
    INNER,
    LEFT, // Left outer join
    // Potentially RIGHT and FULL_OUTER in the future
}
