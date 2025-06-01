package com.example.dsel.ingestion.schema;

import borg.trikeshed.isam.RecordMeta;
import borg.trikeshed.nio.IOMemento; // Changed from isam.meta to nio

import java.util.ArrayList;
import java.util.List;

public class DselSchemas {

    public static final List<RecordMeta> KLINE_SCHEMA;
    public static final List<RecordMeta> TRADE_SCHEMA;

    private static RecordMeta createMeta(String name, IOMemento type, int defaultSize) {
        // Assuming the constructor based on the prompt's example:
        // new RecordMeta(name, type, beginOffset, endOffset, decoder, encoder, aux)
        // Using -1 for offsets if not applicable or to be determined by ISAM system.
        // The createDecoder/Encoder calls are on the IOMemento enum instances.
        return new RecordMeta(name, type, -1, -1, type.createDecoder(defaultSize), type.createEncoder(defaultSize), null);
    }
    
    private static RecordMeta createStringMeta(String name) {
        // For IoString, size for createDecoder/Encoder might be handled differently (e.g. -1 for variable length)
        // Using -1 as a placeholder for variable size, assuming the IOMemento.IoString methods handle it.
        return new RecordMeta(name, IOMemento.IoString, -1, -1, IOMemento.IoString.createDecoder(-1), IOMemento.IoString.createEncoder(-1), null);
    }

    static {
        KLINE_SCHEMA = new ArrayList<>();
        // BinanceKline: (long openTime, String open, String high, String low, String close, String volume, long closeTime, String quoteAssetVolume, int numberOfTrades, String takerBuyBaseAssetVolume, String takerBuyQuoteAssetVolume, String ignore)
        KLINE_SCHEMA.add(createMeta("openTime", IOMemento.IoLong, 8));
        KLINE_SCHEMA.add(createMeta("open", IOMemento.IoDouble, 8)); // DTO String to Double
        KLINE_SCHEMA.add(createMeta("high", IOMemento.IoDouble, 8)); // DTO String to Double
        KLINE_SCHEMA.add(createMeta("low", IOMemento.IoDouble, 8)); // DTO String to Double
        KLINE_SCHEMA.add(createMeta("close", IOMemento.IoDouble, 8)); // DTO String to Double
        KLINE_SCHEMA.add(createMeta("volume", IOMemento.IoDouble, 8)); // DTO String to Double
        KLINE_SCHEMA.add(createMeta("closeTime", IOMemento.IoLong, 8));
        KLINE_SCHEMA.add(createMeta("quoteAssetVolume", IOMemento.IoDouble, 8)); // DTO String to Double
        KLINE_SCHEMA.add(createMeta("numberOfTrades", IOMemento.IoInt, 4));
        KLINE_SCHEMA.add(createMeta("takerBuyBaseAssetVolume", IOMemento.IoDouble, 8)); // DTO String to Double
        KLINE_SCHEMA.add(createMeta("takerBuyQuoteAssetVolume", IOMemento.IoDouble, 8)); // DTO String to Double
        KLINE_SCHEMA.add(createStringMeta("ignore")); // DTO String to IoString

        TRADE_SCHEMA = new ArrayList<>();
        // BinanceTrade: (long tradeId, String price, String qty, String quoteQty, long time, boolean isBuyerMaker, boolean isBestMatch)
        TRADE_SCHEMA.add(createMeta("tradeId", IOMemento.IoLong, 8));
        TRADE_SCHEMA.add(createMeta("price", IOMemento.IoDouble, 8)); // DTO String to Double
        TRADE_SCHEMA.add(createMeta("qty", IOMemento.IoDouble, 8)); // DTO String to Double
        TRADE_SCHEMA.add(createMeta("quoteQty", IOMemento.IoDouble, 8)); // DTO String to Double
        TRADE_SCHEMA.add(createMeta("time", IOMemento.IoLong, 8));
        TRADE_SCHEMA.add(createMeta("isBuyerMaker", IOMemento.IoBoolean, 1));
        TRADE_SCHEMA.add(createMeta("isBestMatch", IOMemento.IoBoolean, 1));
    }
}
