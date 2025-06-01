package com.example.dsel.ingestion.service;

import com.example.dsel.ingestion.dto.BinanceKline;
import com.example.dsel.ingestion.dto.BinanceTrade;
// No direct schema access needed here if passed as method arg, but good for context.
// import com.example.dsel.ingestion.schema.DselSchemas;
// Imports for Join, Series, RowVec are implicit via D.* static methods if not directly typed.
// For explicit types:
import borg.trikeshed.lib.Series;    // For return types like Series<RowVec>
import borg.trikeshed.cursor.RowVec;   // For List<RowVec>
import borg.trikeshed.lib.Join;     // For List<Join<...>>
import borg.trikeshed.isam.RecordMeta; // Changed ColumnMeta to RecordMeta
import com.yourdomain.bikeshed.dsel.D;
// import borg.trikeshed.isam.meta.IOMemento;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

public class DataTransformerService {

    public Series<RowVec> transformKlines(Series<BinanceKline> apiKlinesSeries, List<RecordMeta> klineSchema) {
        if (apiKlinesSeries == null) {
            throw new IllegalArgumentException("Input klines series must not be null.");
        }
        if (klineSchema == null) {
            throw new IllegalArgumentException("Kline schema must not be null.");
        }
        // For robustness, a check like klineSchema.size() == 12 could be added.
        final int expectedSchemaSize = 12; // Example for klines

        return apiKlinesSeries.alpha(kline -> { // kline is a single BinanceKline DTO
            // Ensure schema size matches expectation, or handle potential IndexOutOfBounds
            if (klineSchema.size() < expectedSchemaSize) {
                 throw new IllegalArgumentException("Kline schema size is less than expected " + expectedSchemaSize);
            }

            @SuppressWarnings("unchecked") // For Join[] array creation
            Join<Object, Supplier<RecordMeta>>[] joinsArray = new Join[expectedSchemaSize];

            joinsArray[0] = D.jn(kline.openTime(), () -> klineSchema.get(0));
            joinsArray[1] = D.jn(Double.parseDouble(kline.open()), () -> klineSchema.get(1));
            joinsArray[2] = D.jn(Double.parseDouble(kline.high()), () -> klineSchema.get(2));
            joinsArray[3] = D.jn(Double.parseDouble(kline.low()), () -> klineSchema.get(3));
            joinsArray[4] = D.jn(Double.parseDouble(kline.close()), () -> klineSchema.get(4));
            joinsArray[5] = D.jn(Double.parseDouble(kline.volume()), () -> klineSchema.get(5));
            joinsArray[6] = D.jn(kline.closeTime(), () -> klineSchema.get(6));
            joinsArray[7] = D.jn(Double.parseDouble(kline.quoteAssetVolume()), () -> klineSchema.get(7));
            joinsArray[8] = D.jn(kline.numberOfTrades(), () -> klineSchema.get(8));
            joinsArray[9] = D.jn(Double.parseDouble(kline.takerBuyBaseAssetVolume()), () -> klineSchema.get(9));
            joinsArray[10] = D.jn(Double.parseDouble(kline.takerBuyQuoteAssetVolume()), () -> klineSchema.get(10));
            joinsArray[11] = D.jn(kline.ignore(), () -> klineSchema.get(11));
            
            return D.rv(joinsArray); // D.rv takes varargs Join[]
        });
    }

    public Series<RowVec> transformTrades(Series<BinanceTrade> apiTradesSeries, List<RecordMeta> tradeSchema) {
        if (apiTradesSeries == null) {
            throw new IllegalArgumentException("Input trades series must not be null.");
        }
        if (tradeSchema == null) {
            throw new IllegalArgumentException("Trade schema must not be null.");
        }
        final int expectedSchemaSize = 7; // Example for trades

        return apiTradesSeries.alpha(trade -> {
             if (tradeSchema.size() < expectedSchemaSize) {
                 throw new IllegalArgumentException("Trade schema size is less than expected " + expectedSchemaSize);
            }

            @SuppressWarnings("unchecked") // For Join[] array creation
            Join<Object, Supplier<RecordMeta>>[] joinsArray = new Join[expectedSchemaSize];

            joinsArray[0] = D.jn(trade.tradeId(), () -> tradeSchema.get(0));
            joinsArray[1] = D.jn(Double.parseDouble(trade.price()), () -> tradeSchema.get(1));
            joinsArray[2] = D.jn(Double.parseDouble(trade.qty()), () -> tradeSchema.get(2));
            joinsArray[3] = D.jn(Double.parseDouble(trade.quoteQty()), () -> tradeSchema.get(3));
            joinsArray[4] = D.jn(trade.time(), () -> tradeSchema.get(4));
            joinsArray[5] = D.jn(trade.isBuyerMaker(), () -> tradeSchema.get(5));
            joinsArray[6] = D.jn(trade.isBestMatch(), () -> tradeSchema.get(6));
            
            return D.rv(joinsArray);
        });
    }
}
