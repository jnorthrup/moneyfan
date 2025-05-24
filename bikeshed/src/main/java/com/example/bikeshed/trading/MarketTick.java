package com.example.bikeshed.trading;

import com.example.bikeshed.dsel.Join;

import java.util.Map;
import java.util.Objects;

/**
 * Represents a single snapshot of market data at a given timestamp.
 * This is the fundamental unit of the market timeline.
 */
public class MarketTick extends Join<Long, Map<String, TickData>> {

    /**
     * Factory method for MarketTick, using the DSEL `jn` glyph.
     * @param timestamp The timestamp of the tick (e.g., epoch milliseconds).
     * @param data A map from asset key (e.g., "BTC_USD") to its `TickData`.
     */
    public MarketTick(Long timestamp, Map<String, TickData> data) {
        super(timestamp, data);
    }

    /**
     * Returns the timestamp of this market tick.
     * Glyph: `timestamp`
     */
    public Long timestamp() {
        return a();
    }

    /**
     * Returns the map of asset keys to their corresponding tick data.
     * Glyph: `data`
     */
    public Map<String, TickData> data() {
        return b();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        MarketTick that = (MarketTick) o;
        return Objects.equals(a(), that.a()) && Objects.equals(b(), that.b());
    }

    @Override
    public int hashCode() {
        return Objects.hash(a(), b());
    }

    @Override
    public String toString() {
        return "MarketTick(timestamp=" + timestamp() + ", data=" + data() + ")";
    }
}
