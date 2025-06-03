package com.vsiwest.moneyfan.strategy;

import com.vsiwest.moneyfan.ingestion.KlineData;
import java.util.List;

public interface TradingStrategy {

    /**
     * Generates a trading signal based on historical data up to a certain point.
     *
     * @param historicalData A list of KlineData representing the historical price and volume data.
     *                       This list could be the complete dataset or a relevant window.
     * @param currentIndex   The index in historicalData that represents the current kline bar
     *                       for which a signal is to be generated. The strategy should typically
     *                       look at data up to (but not including or including, depending on convention)
     *                       this index to make a decision.
     * @return               The trading signal (BUY, SELL, HOLD).
     */
    Signal generateSignal(List<KlineData> historicalData, int currentIndex);

    /**
     * Initializes the strategy with the entire dataset. This can be used by strategies
     * that need to perform some preliminary calculations or setup based on all available data
     * before generating signals iteratively.
     * <p>
     * Implementing classes can override this method if they need such initialization.
     * By default, it does nothing.
     *
     * @param allData The complete list of KlineData.
     */
    default void init(List<KlineData> allData) {
        // Default implementation does nothing.
    }
}
