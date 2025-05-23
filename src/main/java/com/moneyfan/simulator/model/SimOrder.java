package com.moneyfan.simulator.model;

import java.util.Objects;

public record SimOrder(
    AssetKey assetKey,
    OrderSide side, // BUY or SELL
    OrderType type, // MARKET or LIMIT
    double quantity,  // Amount of base asset
    double price,     // Price for LIMIT orders; actual fill price for MARKET
    long timestamp,   // Simulation tick/timestamp when order was created/filled
    String agentId
) {
    public enum OrderSide { BUY, SELL }
    public enum OrderType { MARKET, LIMIT }

    public SimOrder { // Compact constructor validation
        Objects.requireNonNull(assetKey); Objects.requireNonNull(side); Objects.requireNonNull(type); Objects.requireNonNull(agentId);
        if (quantity <= 0) throw new IllegalArgumentException("Quantity must be positive.");
        if (type == OrderType.LIMIT && price <= 0) throw new IllegalArgumentException("Limit price must be positive for LIMIT orders.");
    }
}
