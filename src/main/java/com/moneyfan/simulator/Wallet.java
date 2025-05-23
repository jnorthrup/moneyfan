package com.moneyfan.simulator;

import java.math.BigDecimal;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Represents an agent's wallet, holding balances of different assets.
 * Balances are stored as BigDecimal for precision.
 * This class is immutable; operations return new Wallet instances.
 */
public record Wallet(Map<String, BigDecimal> balances) {

    public Wallet() {
        this(Collections.emptyMap());
    }

    public Wallet(Map<String, BigDecimal> balances) {
        this.balances = Collections.unmodifiableMap(new HashMap<>(Objects.requireNonNull(balances)));
    }

    public BigDecimal getBalance(String assetSymbol) {
        return balances.getOrDefault(assetSymbol, BigDecimal.ZERO);
    }

    public Wallet updateBalance(String assetSymbol, BigDecimal newBalance) {
        if (newBalance.compareTo(BigDecimal.ZERO) < 0) {
            throw new IllegalArgumentException("Balance cannot be negative: " + assetSymbol + " " + newBalance);
        }
        Map<String, BigDecimal> newBalances = new HashMap<>(this.balances);
        if (newBalance.compareTo(BigDecimal.ZERO) == 0) {
            newBalances.remove(assetSymbol);
        } else {
            newBalances.put(assetSymbol, newBalance);
        }
        return new Wallet(newBalances);
    }
}
