package com.moneyfan.dsel.dsel;

/**
 * Represents the OHLCV part of a Kline record using nested Join structures.
 * - openHigh: A Join of Open price (Double) and High price (Double).
 * - lowCloseVolume: A Join where:
 *   - first is another Join of Low price (Double) and Close price (Double).
 *   - second is Volume (Double).
 */
public record KlineValues(
    Join<Double, Double> openHigh, // Open, High
    Join<Join<Double, Double>, Double> lowCloseVolume // ((Low, Close), Volume)
) {
    // Convenience getters could be added here if needed, e.g., getOpen(), getClose()
}
