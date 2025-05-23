package com.moneyfan.data;

/**
 * Represents a single candlestick data point.
 * Fields are based on typical Binance kline data.
 */
public record Candle(
    long openTime,
    double open,
    double high,
    double low,
    double close,
    double volume,
    long closeTime,
    double quoteAssetVolume,
    long numberOfTrades,
    double takerBuyBaseAssetVolume,
    double takerBuyQuoteAssetVolume,
    String ignore // Often a placeholder field in Binance CSV
) {}
