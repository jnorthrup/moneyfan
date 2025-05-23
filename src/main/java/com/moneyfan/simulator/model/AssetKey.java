package com.moneyfan.simulator.model;

import java.util.Objects;

public record AssetKey(String baseAsset, String quoteAsset) implements Comparable<AssetKey> {
    public AssetKey {
        Objects.requireNonNull(baseAsset); Objects.requireNonNull(quoteAsset);
    }
    public static AssetKey of(String pairString) { // e.g., "BTC/USDT"
        String[] parts = pairString.split("/");
        if (parts.length != 2) throw new IllegalArgumentException("Invalid pair string: " + pairString);
        return new AssetKey(parts[0].toUpperCase(), parts[1].toUpperCase());
    }
    public String toSymbol() { return baseAsset + quoteAsset; } // For Binance-like symbols
    public String toPairString() { return baseAsset + "/" + quoteAsset; }
    @Override public String toString() { return toPairString(); }
    @Override public int compareTo(AssetKey o) { return toPairString().compareTo(o.toPairString()); }
}
