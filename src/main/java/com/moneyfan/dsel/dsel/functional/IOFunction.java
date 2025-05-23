package com.moneyfan.dsel.dsel.functional;

import java.io.IOException;

/**
 * A functional interface similar to Function but allowing for checked IOException.
 * @param <T> the type of the input to the function
 * @param <R> the type of the result of the function
 */
@FunctionalInterface
public interface IOFunction<T, R> {
    /**
     * Applies this function to the given argument.
     *
     * @param t the function argument
     * @return the function result
     * @throws IOException if an I/O error occurs
     */
    R apply(T t) throws IOException;
}
