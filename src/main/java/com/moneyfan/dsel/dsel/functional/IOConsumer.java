package com.moneyfan.dsel.dsel.functional;

import java.io.IOException;

/**
 * A functional interface similar to BiConsumer but allowing for checked IOException.
 * @param <T> the type of the first argument to the operation
 * @param <U> the type of the second argument to the operation
 */
@FunctionalInterface
public interface IOConsumer<T, U> {
    /**
     * Performs this operation on the given arguments.
     *
     * @param t the first input argument
     * @param u the second input argument
     * @throws IOException if an I/O error occurs
     */
    void accept(T t, U u) throws IOException;
}
