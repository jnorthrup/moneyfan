package borg.trikeshed.isam; // Changed package

import borg.trikeshed.lib.Join; // Changed import
import borg.trikeshed.nio.IOMemento; // Changed from isam.meta to nio
// import com.yourdomain.bikeshed.type.TypeMemento; // This will be unused if IOMemento replaces it.
import org.jetbrains.annotations.NotNull;

/**
 * Represents metadata for a record field (formerly column), which is a {@link Join} of the field name (String)
 * and its {@link IOMemento} (formerly TypeMemento).
 *
 * In Kotlin: `typealias RecordMeta = Join<String, IOMemento>`
 */
public interface RecordMeta extends Join<String, borg.trikeshed.nio.IOMemento> { // Updated generic type to nio.IOMemento

    /**
     * Factory method to create a RecordMeta instance.
     * @param name The name of the field.
     * @param type The IOMemento describing the field's data type.
     * @return A new RecordMeta instance.
     */
    static @NotNull RecordMeta of(@NotNull String name, @NotNull IOMemento type) { // Renamed return, updated param type
        return new ImmutableRecordMeta(name, type); // Changed to new inner class name
    }

    /**
     * Returns the name of the column.
     * @return The column name.
     */
    default @NotNull String name() {
        return fst();
    }

    /**
     * Returns the TypeMemento of the column.
     * @return The column's TypeMemento.
     */
    default @NotNull IOMemento type() { // Changed return type
        return snd();
    }

    // Inner class for the immutable implementation
    final class ImmutableRecordMeta implements RecordMeta { // Removed "extends borg.trikeshed.lib.Join.ImmutableJoin"
        private final String recordName;
        private final borg.trikeshed.nio.IOMemento recordType; // Changed to nio.IOMemento

        private ImmutableRecordMeta(String name, borg.trikeshed.nio.IOMemento type) { // Changed to nio.IOMemento
            this.recordName = java.util.Objects.requireNonNull(name, "name must not be null");
            this.recordType = java.util.Objects.requireNonNull(type, "type must not be null");
        }

        @Override
        public String fst() {
            return this.recordName;
        }

        @Override
        public borg.trikeshed.nio.IOMemento snd() { // Changed to nio.IOMemento
            return this.recordType;
        }
        // Default methods name() and type() in RecordMeta will use fst() and snd()
    }
}
