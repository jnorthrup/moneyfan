package com.moneyfan.binance;

import com.moneyfan.core.IOMemento;
import com.moneyfan.core.Scalar;
import com.moneyfan.grid.GridCursor;
import com.moneyfan.io.CSVCursorReader;
import com.moneyfan.io.ISAMWriter;

import java.nio.file.Path;
import java.util.List;

/**
 * Utility for ingesting Binance kline CSV into ISAM format.
 */
public final class BinanceIngest {

    private BinanceIngest() {}

    public static void convertCsvToIsam(Path csvPath, Path isamPath) throws Exception {
        List<Scalar> schema = klineSchema();
        GridCursor grid = CSVCursorReader.read(csvPath, schema);
        ISAMWriter.write(grid, isamPath);
    }

    private static List<Scalar> klineSchema() {
        return List.of(
                Scalar.of(IOMemento.IO_LONG, "open_time"),
                Scalar.of(IOMemento.IO_DOUBLE, "open"),
                Scalar.of(IOMemento.IO_DOUBLE, "high"),
                Scalar.of(IOMemento.IO_DOUBLE, "low"),
                Scalar.of(IOMemento.IO_DOUBLE, "close"),
                Scalar.of(IOMemento.IO_DOUBLE, "volume"),
                Scalar.of(IOMemento.IO_LONG, "close_time"),
                Scalar.of(IOMemento.IO_DOUBLE, "quote_asset_volume"),
                Scalar.of(IOMemento.IO_INT, "number_of_trades"),
                Scalar.of(IOMemento.IO_DOUBLE, "taker_buy_base_volume"),
                Scalar.of(IOMemento.IO_DOUBLE, "taker_buy_quote_volume")
        );
    }
}