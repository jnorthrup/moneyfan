package com.vsiwest.dsel.ingestion.service;

import com.example.dsel.ingestion.dto.BinanceKline;
import com.example.dsel.ingestion.dto.BinanceTrade;
// No direct schema access needed here if passed as method arg, but good for context.
// import com.example.dsel.ingestion.schema.DselSchemas; 
import com.vsiwest.bikeshed.core.Join;
import com.vsiwest.bikeshed.core.Series;
import com.vsiwest.bikeshed.core.RowVec;
import com.vsiwest.bikeshed.type.ColumnMeta; // Changed from RecordMeta
import com.vsiwest.bikeshed.dsel.D; // Added DSEL import
// IOMemento might not be directly needed here if schema drives types, but good for context.
// import borg.trikeshed.isam.meta.IOMemento; 

import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

public class DataTransformerService {

    public Series<RowVec> transformKlines(List<BinanceKline> apiKlines, List<ColumnMeta> klineSchema) { // RecordMeta -> ColumnMeta
        if (apiKlines == null) {
            throw new IllegalArgumentException("Input klines list must not be null.");
        }
        if (klineSchema == null) {
            throw new IllegalArgumentException("Kline schema must not be null.");
        }
        // Assuming klineSchema.size() is validated by the caller or implicitly correct.
        // For robustness, a check like klineSchema.size() == 12 could be added if 12 is a fixed number of fields.

        List<RowVec> rowVecs = new ArrayList<>(apiKlines.size());

        for (BinanceKline kline : apiKlines) {
            List<Join<Object, Supplier<ColumnMeta>>> joins = new ArrayList<>(klineSchema.size()); // RecordMeta -> ColumnMeta

            // Order of fields in DTO must match order in klineSchema from DselSchemas
            joins.add(D.jn(kline.openTime(), () -> klineSchema.get(0))); // Join.of -> D.jn
            joins.add(D.jn(Double.parseDouble(kline.open()), () -> klineSchema.get(1)));
            joins.add(D.jn(Double.parseDouble(kline.high()), () -> klineSchema.get(2)));
            joins.add(D.jn(Double.parseDouble(kline.low()), () -> klineSchema.get(3)));
            joins.add(D.jn(Double.parseDouble(kline.close()), () -> klineSchema.get(4)));
            joins.add(D.jn(Double.parseDouble(kline.volume()), () -> klineSchema.get(5)));
            joins.add(D.jn(kline.closeTime(), () -> klineSchema.get(6)));
            joins.add(D.jn(Double.parseDouble(kline.quoteAssetVolume()), () -> klineSchema.get(7)));
            joins.add(D.jn(kline.numberOfTrades(), () -> klineSchema.get(8)));
            joins.add(D.jn(Double.parseDouble(kline.takerBuyBaseAssetVolume()), () -> klineSchema.get(9)));
            joins.add(D.jn(Double.parseDouble(kline.takerBuyQuoteAssetVolume()), () -> klineSchema.get(10)));
            joins.add(D.jn(kline.ignore(), () -> klineSchema.get(11)));
            
            rowVecs.add(D.rv(joins)); // RowVec.of(joins.toArray(new Join[0])) -> D.rv(joins)
        }
        return D.sr(rowVecs.size(), rowVecs::get); // Series.of -> D.sr
    }

    public Series<RowVec> transformTrades(List<BinanceTrade> apiTrades, List<ColumnMeta> tradeSchema) { // RecordMeta -> ColumnMeta
        if (apiTrades == null) {
            throw new IllegalArgumentException("Input trades list must not be null.");
        }
        if (tradeSchema == null) {
            throw new IllegalArgumentException("Trade schema must not be null.");
        }
        // Assuming tradeSchema.size() is validated by the caller or implicitly correct.

        List<RowVec> rowVecs = new ArrayList<>(apiTrades.size());

        for (BinanceTrade trade : apiTrades) {
            List<Join<Object, Supplier<ColumnMeta>>> joins = new ArrayList<>(tradeSchema.size()); // RecordMeta -> ColumnMeta

            // Order of fields in DTO must match order in tradeSchema from DselSchemas
            joins.add(D.jn(trade.tradeId(), () -> tradeSchema.get(0))); // Join.of -> D.jn
            joins.add(D.jn(Double.parseDouble(trade.price()), () -> tradeSchema.get(1)));
            joins.add(D.jn(Double.parseDouble(trade.qty()), () -> tradeSchema.get(2)));
            joins.add(D.jn(Double.parseDouble(trade.quoteQty()), () -> tradeSchema.get(3)));
            joins.add(D.jn(trade.time(), () -> tradeSchema.get(4)));
            joins.add(D.jn(trade.isBuyerMaker(), () -> tradeSchema.get(5)));
            joins.add(D.jn(trade.isBestMatch(), () -> tradeSchema.get(6)));
            
            rowVecs.add(D.rv(joins)); // RowVec.of(joins.toArray(new Join[0])) -> D.rv(joins)
        }
        return D.sr(rowVecs.size(), rowVecs::get); // Series.of -> D.sr
    }
}
