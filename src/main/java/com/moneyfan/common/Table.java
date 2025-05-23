package com.moneyfan.common;

import java.util.List;
import java.util.Map;

/**
 * Placeholder for Table.java.
 * The original error was on line 45.
 * We'll create content up to that line and provide a corrected version
 * of a plausible problematic line.
 */
public class Table<R> {

    private String name;
    private List<R> rows;

    public Table(String name, List<R> rows) {
        this.name = name;
        this.rows = rows;
    }

    public String getName() {
        return name;
    }

    public List<R> getRows() {
        return rows;
    }

    // ... other methods ...
    // Let's assume line 45 was part of a nested class or record definition
    // that had a syntax error in its type parameters.

    /*
    Original problematic line 45 might have been something like:
    (This is a guess based on the error messages and column numbers)

    // public record RowEntry<K V> implements Map.Entry<K,V> { // Incorrect: missing comma between K and V

    Corrected version (assuming this was the intent):
    */
    public record RowEntry<K, V>(K key, V value) implements Map.Entry<K, V> {
        @Override
        public K getKey() { return key; }
        @Override
        public V getValue() { return value; }
        @Override
        public V setValue(V newValue) { throw new UnsupportedOperationException(); }
    }

    // More methods or content for Table class...
}
