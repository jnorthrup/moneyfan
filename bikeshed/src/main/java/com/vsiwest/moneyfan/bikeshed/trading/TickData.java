package com.vsiwest.moneyfan.bikeshed.trading;

import com.vsiwest.moneyfan.bikeshed.core.Join;
import com.vsiwest.moneyfan.bikeshed.dsel.D; // Assuming D.jn is used for Join creation

import java.util.Objects;

/**
 * Represents a single market tick (e.g., OHLCV data).
 * This is structured as a nested Join to demonstrate the DSEL's compositional nature.
 * Specifically, it's a Join of (Open, Join(High, Join(Low, Join(Close, Volume)))).
 */
public class TickData extends Join<Double, Join<Double, Join<Double, Join<Double, Double>>>> {

    /**
     * Factory method for TickData, using the DSEL `jn` glyph.
     * @param open Open price.
     * @param high High price.
     * @param low Low price.
     * @param close Close price.
     * @param volume Volume traded.
     */
    public static TickData of(Double open, Double high, Double low, Double close, Double volume) {
        // Using D.jn for concise Join construction
        return new TickData(D.jn(open, D.jn(high, D.jn(low, D.jn(close, volume)))));
    }

    // Private constructor to enforce factory method usage
    private TickData(Join<Double, Join<Double, Join<Double, Join<Double, Double>>>> nestedJoin) {
        super(nestedJoin.first(), nestedJoin.second());
    }

    // Accessor methods for convenience
    public Double getOpen() {
        return first();
    }

    public Double getHigh() {
        return second().first();
    }

    public Double getLow() {
        return second().second().first();
    }

    public Double getClose() {
        return second().second().second().first();
    }

    public Double getVolume() {
        return second().second().second().second();
    }

    @Override
    public String toString() {
        return "TickData{" +
               "open=" + getOpen() +
               ", high=" + getHigh() +
               ", low=" + getLow() +
               ", close=" + getClose() +
               ", volume=" + getVolume() +
               '}';
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        TickData tickData = (TickData) o;
        return Objects.equals(getOpen(), tickData.getOpen()) &&
               Objects.equals(getHigh(), tickData.getHigh()) &&
               Objects.equals(getLow(), tickData.getLow()) &&
               Objects.equals(getClose(), tickData.getClose()) &&
               Objects.equals(getVolume(), tickData.getVolume());
    }

    @Override
    public int hashCode() {
        return Objects.hash(getOpen(), getHigh(), getLow(), getClose(), getVolume());
    }
}
