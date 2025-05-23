package com.moneyfan.dsl.typeevidence;

import com.moneyfan.dsl.Join;

/**
 * Represents a token with a specific semantic type, value, and name.
 * This can be viewed as a composition: Join<NamedValue, Type>, where NamedValue is Join<Name, Value>.
 * Or, as used here: A record holding these three pieces of information directly.
 *
 * @param <V> The type of the token's value.
 */
public record SemanticToken<V>(String name, V value, SemanticTokenType type) {

    /**
     * Glyph shorthand factory method.
     * st(name, value, type)
     */
    public static <V> SemanticToken<V> st(String name, V value, SemanticTokenType type) {
        return new SemanticToken<>(name, value, type);
    }
}
