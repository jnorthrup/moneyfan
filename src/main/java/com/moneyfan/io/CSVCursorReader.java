package com.moneyfan.io;

import com.moneyfan.core.CellMeta;
import com.moneyfan.core.Scalar;
import com.moneyfan.grid.Cell;
import com.moneyfan.grid.GridCursor;
import com.moneyfan.grid.RowVec;
import com.moneyfan.grid.Vect0r;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

/**
 * Very simple CSV reader that converts rows into an in-memory GridCursor.
 * Assumes delimiter is comma and no escaping/quotes complexities.
 */
public final class CSVCursorReader {

    private CSVCursorReader() {}

    public static GridCursor read(Path csvPath, List<Scalar> schema) throws IOException {
        List<RowVec> rows = new ArrayList<>();
        List<String> lines = Files.readAllLines(csvPath);
        for(String line: lines) {
            if(line.isBlank()) continue;
            String[] tokens = line.split(",");
            if(tokens.length!=schema.size())
                throw new IOException("CSV column count mismatch on line: " + line);
            List<Cell> cells = new ArrayList<>(schema.size());
            for(int i=0;i<schema.size();i++) {
                Scalar sc = schema.get(i);
                Object parsed = parseToken(tokens[i], sc);
                Supplier<Scalar> supplier = () -> sc;
                cells.add(new Cell(parsed, new CellMeta(supplier)));
            }
            rows.add(new RowVec(Vect0r.fromList(cells)));
        }
        return new GridCursor(Vect0r.fromList(rows));
    }

    private static Object parseToken(String token, Scalar scalar) {
        return switch (scalar.type()) {
            case IO_INT -> Integer.parseInt(token);
            case IO_LONG -> Long.parseLong(token);
            case IO_DOUBLE -> Double.parseDouble(token);
            case IO_LOCAL_DATE -> java.time.LocalDate.parse(token);
            case IO_INSTANT -> java.time.Instant.parse(token);
            case IO_STRING_FIXED -> token; // no padding removal yet
        };
    }
}