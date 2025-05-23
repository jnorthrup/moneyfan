package com.moneyfan.simulator.model;

public enum AssetMutation {
    // Action Type (Primary Decision)
    HOLD_ACTION(0.0), // No change, or explicitly hold
    BUY_ACTION(1.0),  // Signal to buy
    SELL_ACTION(2.0), // Signal to sell

    // Order Type (How to execute)
    AS_MARKET_ORDER(1.0), // Execute at current market price
    AS_LIMIT_ORDER(2.0),  // Execute at a specific limit price or better

    // Parameters (Modifiers for the action)
    PRICE_FRACTION(1.0),  // For limit orders: price as fraction/multiple of current close. e.g., 0.99 for 1% below, 1.01 for 1% above. For market, can be ignored or be 1.0.
    QUANTITY_FRACTION(0.1); // Fraction of available quote (for buy) or base (for sell) currency to use. e.g., 0.5 for 50%.

    public final double defaultValue; // A potential default or neutral value for an agent's output
    AssetMutation(double defaultValue) { this.defaultValue = defaultValue; }

    // Consider adding a method to get all relevant mutations for an agent's output array size.
    public static final AssetMutation[] ALL_MUTATIONS = values();
    public static final int OUTPUT_SIZE = ALL_MUTATIONS.length;
}
