package com.example.bikeshed.trading;

import com.example.bikeshed.dsel.Join;

import java.util.Collections;
import java.util.Map;
import java.util.Objects;

/**
 * Represents the state of an agent's wallet, e.g., balances of different assets.
 * Uses `Join` for immutability.
 */
public class WalletState extends Join<Map<String, Double>, Void> { // Void as placeholder for future extensions

    /**
     * Factory method for WalletState, using the DSEL `jn` glyph.
     *
     * @param balances A map from asset symbol (e.g., "BTC", "USD") to its balance.
     */
    public WalletState(Map<String, Double> balances) {
        super(Collections.unmodifiableMap(balances), null);
    }

    /**
     * Returns an unmodifiable map of asset balances.
     * Glyph: `balances`
     */
    public Map<String, Double> balances() {
        return a();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        WalletState that = (WalletState) o;
        return Objects.equals(balances(), that.balances());
    }

    @Override
    public int hashCode() {
        return Objects.hash(balances());
    }

    @Override
    public String toString() {
        return "WalletState(balances=" + balances() + ")";
    }
}
