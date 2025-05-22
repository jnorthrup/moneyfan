package com.moneyfan.grid;

import com.moneyfan.core.IOMemento;
import com.moneyfan.core.Scalar;

import java.util.Objects;

/**
 * Immutable view of a single row in a {@link GridCursor}.  Uses a lazy vector of {@link Cell}.
 */
public record RowVec(Vect0r<Cell> cells) {

    public RowVec {
        Objects.requireNonNull(cells, "cells");
    }

    public int size() {return cells.size();}

    public Cell cell(int colIndex) {return cells.get(colIndex);} // Index checks delegated

    public Object value(int colIndex) {return cells.get(colIndex).value();}

    public Scalar scalar(int colIndex) {return cells.get(colIndex).meta().scalar();}

    public IOMemento type(int colIndex) {return scalar(colIndex).type();}
}