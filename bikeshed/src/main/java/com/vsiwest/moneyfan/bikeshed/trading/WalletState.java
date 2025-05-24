package com.vsiwest.moneyfan.bikeshed.trading;

import com.vsiwest.moneyfan.bikeshed.core.Join;
import com.vsiwest.moneyfan.bikeshed.dsel.D; // Assuming D.jn is used for Join creation

import java.util.Collections;
import java.util.Map;
import java.util.Objects;

/**
 * Represents the state of a trading wallet, holding balances for various assets.
 * This is structured as a Join of (Map<String, Double>, Void) where Void is a placeholder
 * for future extensions (e.g., transaction history, PnL).
 */
public class WalletState extends Join<Map<String, Double>, Void> { // Void as placeholder for future extensions

    /**
     * Factory method for WalletState, using the DSEL `jn` glyph.
     *
     * @param balances A map from asset symbol (e.g., "BTC", "USD") to its balance.
     */
    public WalletState(Map<String, Double> balances) {
        // Use D.jn for concise Join construction, ensuring the map is unmodifiable.
        super(Collections.unmodifiableMap(balances), null);
    }

    /**
     * Returns an unmodifiable map of asset balances.
     *
     * @return A map from asset symbol to its balance.
     */
    public Map<String, Double> balances() {
        return first(); // Using the 'first' accessor from Join
    }

    @Override
    public String toString() {
        return "WalletState{" +
               "balances=" + balances() +
               '}';
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
}
