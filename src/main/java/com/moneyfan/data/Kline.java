package com.moneyfan.data;

import java.math.BigDecimal;

/**
 * Represents a single Kline (candlestick).
 * Based on typical Binance kline data structure.
 */
public record Kline(
    String symbol,
    long openTime,
    BigDecimal open,
    BigDecimal high,
    BigDecimal low,
    BigDecimal close,
    BigDecimal volume,
    long closeTime,
    BigDecimal quoteAssetVolume,
    long numberOfTrades,
    BigDecimal takerBuyBaseAssetVolume,
    BigDecimal takerBuyQuoteAssetVolume,
    String ignore
) {}
