package com.moneyfan.dsl.row;

import com.moneyfan.dsl.typeevidence.SchemaEvidence;
import java.util.Arrays;
import java.util.Objects;

/**
 * A concrete implementation of {@link Row} backed by an array of objects.
 * Assumes values are aligned with the provided {@link SchemaEvidence}.
 */
public class DataRow implements Row {
    private final SchemaEvidence schema;
    private final Object[] values;

    public DataRow(SchemaEvidence schema, Object[] values) {
        Objects.requireNonNull(schema, "Schema cannot be null for DataRow");
        Objects.requireNonNull(values, "Values cannot be null for DataRow");
        if (schema.size() != values.length) {
            throw new IllegalArgumentException(
                "Schema size (" + schema.size() + ") must match values array length (" + values.length + ")"
            );
        }
        this.schema = schema;
        // Defensive copy to maintain immutability if the input array is mutable
        this.values = Arrays.copyOf(values, values.length);
    }

    @Override
    @SuppressWarnings("unchecked")
    public <T> T get(String columnName) {
        int index = schema.indexOf(columnName);
        if (index == -1) {
            throw new IllegalArgumentException("Column not found: " + columnName + " in schema: " + schema.schemaName());
        }
        return (T) values[index]; // Type safety relies on correct schema and population
    }

    @Override
    @SuppressWarnings("unchecked")
    public <T> T get(int columnIndex) {
        if (columnIndex < 0 || columnIndex >= values.length) {
            throw new ArrayIndexOutOfBoundsException("Column index " + columnIndex + " out of bounds for schema size " + values.length);
        }
        return (T) values[columnIndex]; // Type safety relies on correct schema and population
    }

    @Override
    public SchemaEvidence schema() {
        return schema;
    }

    @Override
    public Object[] getValues() {
        return Arrays.copyOf(values, values.length); // Return a copy for encapsulation
    }

    @Override
    public String toString() {
        return "DataRow{" + "schema=" + schema.schemaName() + ", values=" + Arrays.toString(values) + '}';
    }
}
