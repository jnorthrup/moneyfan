package com.moneyfan.grid;

/**
 * Represents a row within a grid. Essentially a vector of {@link Cell}.
 */
public record RowVec(Vect0r<Cell> cells) {
}