package com.moneyfan.storage;

import com.moneyfan.common.Join;
import com.moneyfan.model.IsamMeta;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * A simplified ISAM (Indexed Sequential Access Method) storage class.
 * This is a mock implementation for demonstration purposes.
 *
 * @param <T> The type of {@link Join} records to store.
 */
public class IsamStorage<T extends Join<?, ?>> implements AutoCloseable {

    private final IsamMeta meta;
    private final List<T> dataStore;
    private boolean isOpen;

    public IsamStorage(IsamMeta meta) {
        this.meta = meta;
        this.dataStore = new ArrayList<>();
        this.isOpen = true;
        System.out.println("IsamStorage initialized for: " + meta.getDescription());
    }

    public void write(List<T> data) throws IOException {
        if (!isOpen) throw new IOException("Storage is closed.");
        System.out.println("Writing " + data.size() + " records to ISAM for " + meta.getDescription());
        this.dataStore.addAll(data);
    }

    public List<T> read() throws IOException {
        if (!isOpen) throw new IOException("Storage is closed.");
        System.out.println("Reading " + dataStore.size() + " records from ISAM for " + meta.getDescription());
        return new ArrayList<>(this.dataStore);
    }

    @Override
    public void close() throws IOException {
        isOpen = false;
        System.out.println("IsamStorage closed for: " + meta.getDescription());
    }
}
