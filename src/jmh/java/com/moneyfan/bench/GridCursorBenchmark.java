package com.moneyfan.bench;

import com.moneyfan.core.IOMemento;
import com.moneyfan.core.Scalar;
import com.moneyfan.grid.*;
import org.openjdk.jmh.annotations.*;

import java.util.List;
import java.util.concurrent.TimeUnit;

@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.SECONDS)
@State(Scope.Benchmark)
public class GridCursorBenchmark {

    private GridCursor grid;

    @Setup
    public void setup() {
        List<Scalar> schema = List.of(
                Scalar.of(IOMemento.IO_INT, "id"),
                Scalar.of(IOMemento.IO_DOUBLE, "value")
        );
        int rows = 100_000;
        Vect0r<RowVec> rowVec = Vect0r.of(rows, i -> makeRow(i, i*0.1, schema));
        grid = new GridCursor(rowVec);
    }

    private RowVec makeRow(int id, double value, List<Scalar> schema) {
        List<Cell> cells = List.of(
                new Cell(id, new com.moneyfan.core.CellMeta(() -> schema.get(0))),
                new Cell(value, new com.moneyfan.core.CellMeta(() -> schema.get(1)))
        );
        return new RowVec(Vect0r.fromList(cells));
    }

    @Benchmark
    public double sumValuesSelect() {
        GridCursor selected = grid.select("value");
        double sum = 0.0;
        for(RowVec row: selected.rows()) {
            sum += (Double) row.get(0).value();
        }
        return sum;
    }
}