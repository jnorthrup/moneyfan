package com.moneyfan.dsl.typeevidence;

/**
 * Represents the evidence for a single field's type, including its name, semantic type, and Java class.
 * This is an immutable structure, akin to a specialized tuple.
 */
public record FieldTypeEvidence(String fieldName, SemanticTokenType semanticType, Class<?> javaType) {

    /**
     * Glyph shorthand factory method.
     * fte(name, semanticType, javaClass)
     */
    public static FieldTypeEvidence fte(String fieldName, SemanticTokenType type, Class<?> javaType) {
        return new FieldTypeEvidence(fieldName, type, javaType);
    }

    @Override
    public String toString() {
        return "Field(" + fieldName + ": " + semanticType + " [" + javaType.getSimpleName() + "])";
    }
}
