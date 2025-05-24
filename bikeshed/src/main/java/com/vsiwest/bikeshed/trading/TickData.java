package com.vsiwest.bikeshed.trading;

import com.example.bikeshed.dsel.Join;
import com.example.bikeshed.dsel.D;

import java.util.Objects;

/**
 * Represents fundamental OHLCV (Open, High, Low, Close, Volume) data for a single asset.
 * Could be extended with Bid/Ask, etc.
 * Uses nested `Join`s for compositional immutability.
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
    public TickData(Double open, Double high, Double low, Double close, Double volume) {
        super(open, D.jn(high, D.jn(low, D.jn(close, volume))));
    }

    /**
     * Open price.
     * Glyph: `open`
     */
    public Double open() {
        return a();
    }

    /**
     * High price.
     * Glyph: `high`
     */
    public Double high() {
        return b().a();
    }

    /**
     * Low price.
     * Glyph: `low`
     */
    public Double low() {
        return b().b().a();
    }

    /**
     * Close price.
     * Glyph: `close`
     */
    public Double close() {
        return b().b().b().a();
    }

    /**
     * Volume traded.
     * Glyph: `volume`
     */
    public Double volume() {
        return b().b().b().b();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        TickData tickData = (TickData) o;
        return Objects.equals(open(), tickData.open()) &&
               Objects.equals(high(), tickData.high()) &&
               Objects.equals(low(), tickData.low()) &&
               Objects.equals(close(), tickData.close()) &&
               Objects.equals(volume(), tickData.volume());
    }

    @Override
    public int hashCode() {
        return Objects.hash(open(), high(), low(), close(), volume());
    }

    @Override
    public String toString() {
        return "TickData(" +
               "open=" + open() +
               ", high=" + high() +
               ", low=" + low() +
               ", close=" + close() +
               ", volume=" + volume() +
               ')';
    }
}
