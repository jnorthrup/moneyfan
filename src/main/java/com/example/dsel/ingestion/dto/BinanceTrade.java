package com.example.dsel.ingestion.dto;

public record BinanceTrade(
    long tradeId,
    String price,
    String qty,
    String quoteQty,
    long time,
    boolean isBuyerMaker,
    boolean isBestMatch
) {}
