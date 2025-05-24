package com.example.dsel.ingestion.dto;

// String fields for numbers that will be handled as BigDecimal in the transformer.
public record BinanceKline(
    long openTime,
    String open,
    String high,
    String low,
    String close,
    String volume,
    long closeTime,
    String quoteAssetVolume,
    int numberOfTrades,
    String takerBuyBaseAssetVolume,
    String takerBuyQuoteAssetVolume,
    String ignore
) {}
