package com.moneyfan.dsl.typeevidence;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * A basic, immutable implementation of {@link SchemaEvidence}.
 */
public class BasicSchemaEvidence implements SchemaEvidence {
    private final String schemaName;
    private final List<FieldTypeEvidence> fieldEvidences;
    private final Map<String, Integer> nameToIndexMap; // For quick lookups by name

    public BasicSchemaEvidence(String schemaName, List<FieldTypeEvidence> fieldEvidences) {
        this.schemaName = schemaName != null ? schemaName : "AnonymousSchema";
        this.fieldEvidences = Collections.unmodifiableList(List.copyOf(fieldEvidences)); // Ensure immutability

        this.nameToIndexMap = IntStream.range(0, this.fieldEvidences.size())
            .boxed()
            .collect(Collectors.toUnmodifiableMap(
                i -> this.fieldEvidences.get(i).fieldName(),
                i -> i,
                (existing, replacement) -> existing // In case of duplicate field names, keep the first one
            ));
    }

    public BasicSchemaEvidence(List<FieldTypeEvidence> fieldEvidences) {
        this("AnonymousSchema", fieldEvidences);
    }

    @Override
    public String schemaName() {
        return schemaName;
    }

    @Override
    public List<FieldTypeEvidence> fields() {
        return fieldEvidences; // Already unmodifiable
    }

    @Override
    public Optional<FieldTypeEvidence> field(String name) {
        Integer index = nameToIndexMap.get(name);
        return (index != null) ? Optional.of(fieldEvidences.get(index)) : Optional.empty();
    }

    @Override
    public Optional<FieldTypeEvidence> field(int index) {
        return (index >= 0 && index < fieldEvidences.size()) ? Optional.of(fieldEvidences.get(index)) : Optional.empty();
    }

    @Override
    public int indexOf(String name) {
        return nameToIndexMap.getOrDefault(name, -1);
    }

    @Override
    public int size() {
        return fieldEvidences.size();
    }
    @Override
    public boolean hasField(String name) { return nameToIndexMap.containsKey(name); }
}
