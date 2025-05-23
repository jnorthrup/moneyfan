package com.moneyfan.dsl.typeevidence;

import java.util.List;
import java.util.Optional;

/**
 * Interface for providing schema information about a {@link com.moneyfan.dsl.row.Row} or a collection of rows.
 * A schema could be conceptualized as Join<SchemaName, List<FieldTypeEvidence>>.
 */
public interface SchemaEvidence {
    String schemaName();
    List<FieldTypeEvidence> fields();
    Optional<FieldTypeEvidence> field(String name);
    Optional<FieldTypeEvidence> field(int index);

    int indexOf(String name); // Returns -1 if not found
    int size(); // Number of fields

    boolean hasField(String name);
}
