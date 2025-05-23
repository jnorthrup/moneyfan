package com.moneyfan.data;

public enum OrderType {
    LIMIT,             // Standard limit order
    MARKET,            // Market order
    STOP_LOSS,         // Stop-loss order (typically becomes a market order when triggered)
    STOP_LOSS_LIMIT,   // Stop-loss order that becomes a limit order when triggered
    TAKE_PROFIT,       // Take-profit order (typically becomes a market order)
    TAKE_PROFIT_LIMIT, // Take-profit order that becomes a limit order
    LIMIT_MAKER        // A limit order that will only be accepted if it can be a maker order (post-only)
}
