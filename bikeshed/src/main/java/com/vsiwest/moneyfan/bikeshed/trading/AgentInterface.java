package com.vsiwest.moneyfan.bikeshed.trading;

import com.vsiwest.moneyfan.bikeshed.dsel.Cursor;
import org.jetbrains.annotations.NotNull;

/**
 * Represents a trading agent that operates on market data.
 * Agents interact with the DSEL through {@link Cursor} for data access.
 * The output is an array of Doubles, representing actions (e.g., allocation for different assets).
 */
public interface AgentInterface {

    /**
     * Decides an action based on the current market observation.
     * Agents receive a {@link Cursor} which provides a view of the shared, immutable
     * market timeline.
     *
     * @param currentMarketData A Cursor providing market data for the agent's lookback period.
     * @return A double array representing the agent's action (e.g., asset allocations).
     */
    @NotNull double[] decideAction(@NotNull Cursor currentMarketData);

}
