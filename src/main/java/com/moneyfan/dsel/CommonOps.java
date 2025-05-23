package com.moneyfan.dsel;

import java.util.function.Function;
import java.util.function.Predicate;

/**
 * Enum bags for common Predicates and Functions.
 */
public enum CommonOps {; // Utility class, not instantiable via constructor

    public enum Predicates {
        IS_POSITIVE_INTEGER(obj -> obj instanceof Integer && (Integer) obj > 0),
        IS_NON_EMPTY_STRING(obj -> obj instanceof String && !((String) obj).isEmpty()),
        IS_NULL(obj -> obj == null),
        IS_NOT_NULL(obj -> obj != null);

        private final Predicate<Object> predicate;
        Predicates(Predicate<Object> predicate) { this.predicate = predicate; }
        @SuppressWarnings("unchecked")
        public <T> Predicate<T> get() { return (Predicate<T>) predicate; }
    }

    public enum Functions {
        TO_STRING_SAFE(obj -> obj == null ? "null" : obj.toString()),
        STRING_TO_LENGTH(obj -> (obj instanceof String) ? ((String)obj).length() : -1),
        IDENTITY(Function.identity());


        private final Function<Object, ?> function;
        Functions(Function<Object, ?> function) { this.function = function; }
        
        @SuppressWarnings("unchecked")
        public <T, R> Function<T, R> get() { return (Function<T, R>) function; }
    }
}
