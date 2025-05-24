package com.yourdomain.bikeshed.trading;

import org.jetbrains.annotations.NotNull;

/**
 * Represents a single market data tick for a specific asset pair at a given timestamp.
 * This is a foundational data structure for the trading simulator.
 */
public class MarketTick {
    private final long timestamp;
    private final @NotNull String symbol; // E.g., "BTCUSDT"
    private final double open;
    private final double high;
    private final double low;
    private final double close;
    private final double volume;

    public MarketTick(long timestamp, @NotNull String symbol, double open, double high, double low, double close, double volume) {
        this.timestamp = timestamp;
        this.symbol = symbol;
        this.open = open;
        this.high = high;
        this.low = low;
        this.close = close;
        this.volume = volume;
    }

    public long getTimestamp() {
        return timestamp;
    }

    public @NotNull String getSymbol() {
        return symbol;
    }

    public double getOpen() {
        return open;
    }

    public double getHigh() {
        return high;
    }

    public double getLow() {
        return low;
    }

    public double getClose() {
        return close;
    }

    public double getVolume() {
        return volume;
    }

    @Override
    public String toString() {
        return "MarketTick{" +
               "timestamp=" + timestamp +
               ", symbol='" + symbol + '\'' +
               ", open=" + open +
               ", high=" + high +
               ", low=" + low +
               ", close=" + close +
               ", volume=" + volume +
               '}';
    }
}
