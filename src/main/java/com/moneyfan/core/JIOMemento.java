package com.moneyfan.core;

/**
 * An enum representing different states or mementos for I/O operations in the 2D grid DSL.
 * This is used for managing state transitions and ensuring immutability.
 */
public enum JIOMemento {
    INITIAL("Initial state before any operation"),
    LOADING("State during data loading"),
    PROCESSING("State during data processing"),
    SAVING("State during data saving"),
    COMPLETED("State after operation completion"),
    ERROR("State indicating an error occurred");
    
    private final String description;
    
    JIOMemento(String description) {
        this.description = description;
    }
    
    public String getDescription() {
        return description;
    }
}