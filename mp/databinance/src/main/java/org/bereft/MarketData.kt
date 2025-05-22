package org.bereft

/**
 * Minimal immutable view of OHLCV market information.
 * Times are Unix epoch milliseconds (UTC).
 */
interface MarketData {
    val openTime: Long
    val open: Double
    val high: Double
    val low: Double
    val close: Double
    val volume: Double
    val closeTime: Long
}