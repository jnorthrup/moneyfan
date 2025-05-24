package com.example.bikeshed.trading;

import com.example.bikeshed.dsel.Cursor;

/**
 * Interface for a trading agent. Agents operate on a shared, immutable timeline (Cursor of ticks).
 */
public interface AgentInterface {
    /**
     * Decides an action based on the current market observation.
     *
     * @param observation The agent's current market observation (a DSEL Cursor).
     * @return An array of doubles representing the agent's actions (e.g., asset allocations).
     */
    double[] decideAction(Cursor observation);

    /**
     * Updates the agent's internal state based on execution results.
     * @param result The outcome of the trade execution.
     */
    void learnFromExecution(ExecutionResult result);
}
