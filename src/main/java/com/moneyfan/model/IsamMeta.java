package com.moneyfan.model;

/**
 * Metadata for ISAM storage.
 * This class would typically hold information about the structure, types,
 * and configuration of an ISAM file. For now, it's simple.
 */
public class IsamMeta {
    private final String description;

    public IsamMeta(String description) {
        this.description = description;
    }

    public String getDescription() {
        return description;
    }
}
