package com.moneyfan.bench;

import com.moneyfan.core.IOMemento;
import com.moneyfan.core.Scalar;
import com.moneyfan.grid.GridCursor;
import com.moneyfan.io.CSVCursorReader;
import com.moneyfan.io.ISAMReader;
import com.moneyfan.io.ISAMWriter;
import org.openjdk.jmh.annotations.*;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.TimeUnit;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Benchmark)
public class CSVvsISAMReadBenchmark {

    @Param({"1000","10000"})
    private int rows;

    private List<Scalar> schema;
    private Path csvPath;
    private Path isamPath;

    @Setup(Level.Trial)
    public void setup() throws Exception {
        schema = List.of(
                Scalar.of(IOMemento.IO_INT, "id"),
                Scalar.of(IOMemento.IO_DOUBLE, "value")
        );
        csvPath = Files.createTempFile("bench", ".csv");
        isamPath = Files.createTempFile("bench", ".bin");
        StringBuilder sb = new StringBuilder();
        for(int i=0;i<rows;i++) {
            sb.append(i).append(',').append(i*0.01).append('\n');
        }
        Files.writeString(csvPath, sb.toString());
        // convert to ISAM once
        GridCursor grid = CSVCursorReader.read(csvPath, schema);
        ISAMWriter.write(grid, isamPath);
    }

    @Benchmark
    public int readCsvAndSum() throws Exception {
        GridCursor grid = CSVCursorReader.read(csvPath, schema);
        int sum = 0;
        for(int i=0;i<grid.rowCount();i++) sum += (Integer) grid.getRow(i).get(0).value();
        return sum;
    }

    @Benchmark
    public int readIsamAndSum() throws Exception {
        try(ISAMReader reader = new ISAMReader(isamPath)) {
            GridCursor grid = reader.open();
            int sum=0;
            for(int i=0;i<grid.rowCount();i++) sum += (Integer) grid.getRow(i).get(0).value();
            return sum;
        }
    }
}