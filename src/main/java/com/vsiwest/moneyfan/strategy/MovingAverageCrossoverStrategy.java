package com.vsiwest.moneyfan.strategy;

import com.vsiwest.moneyfan.ingestion.KlineData;
import java.util.List;
// No need for ArrayList import if not explicitly used for creating new lists within this class

public class MovingAverageCrossoverStrategy implements TradingStrategy {

    private final int shortPeriod;
    private final int longPeriod;

    public MovingAverageCrossoverStrategy(int shortPeriod, int longPeriod) {
        if (shortPeriod <= 0 || longPeriod <= 0) {
            throw new IllegalArgumentException("Periods must be positive.");
        }
        if (shortPeriod >= longPeriod) {
            throw new IllegalArgumentException("Short period must be less than long period.");
        }
        this.shortPeriod = shortPeriod;
        this.longPeriod = longPeriod;
    }

    /**
     * Initializes the strategy. For this particular strategy, no pre-calculation
     * based on the entire dataset is needed, as MAs are calculated on-the-fly.
     *
     * @param allData The complete list of KlineData.
     */
    @Override
    public void init(List<KlineData> allData) {
        // No initialization needed for this strategy
    }

    /**
     * Calculates the Simple Moving Average (SMA) of close prices.
     *
     * @param data      List of KlineData.
     * @param period    The period for the SMA.
     * @param endIndex  The ending index (inclusive) in the data list for the calculation.
     * @return The SMA, or 0.0 if there's not enough data.
     */
    private double calculateSMA(List<KlineData> data, int period, int endIndex) {
        if (endIndex < period - 1 || endIndex >= data.size()) {
            // Not enough data, or endIndex is out of bounds
            return 0.0;
        }

        double sum = 0.0;
        for (int i = endIndex - period + 1; i <= endIndex; i++) {
            sum += data.get(i).getClosePrice();
        }
        return sum / period;
    }

    @Override
    public Signal generateSignal(List<KlineData> historicalData, int currentIndex) {
        // We need at least 'longPeriod' data points to calculate the current long MA,
        // and 'longPeriod' data points up to 'currentIndex - 1' for the previous long MA.
        // Therefore, currentIndex must be at least longPeriod.
        // (currentIndex is 0-based, so longPeriod data points means index goes from 0 to longPeriod-1)
        if (currentIndex < longPeriod) {
            return Signal.HOLD; // Not enough data for the previous long MA
        }

        // Calculate current MAs
        double shortMA_current = calculateSMA(historicalData, shortPeriod, currentIndex);
        double longMA_current = calculateSMA(historicalData, longPeriod, currentIndex);

        // Calculate previous MAs
        // (currentIndex -1 must be valid for shortPeriod and longPeriod)
        double shortMA_previous = calculateSMA(historicalData, shortPeriod, currentIndex - 1);
        double longMA_previous = calculateSMA(historicalData, longPeriod, currentIndex - 1);

        // If any MA calculation returned 0.0 due to insufficient data (which shouldn't happen
        // if currentIndex >= longPeriod, but as a safeguard for calculateSMA returning 0.0)
        if (shortMA_current == 0.0 || longMA_current == 0.0 || shortMA_previous == 0.0 || longMA_previous == 0.0) {
            return Signal.HOLD;
        }

        // Crossover upwards: current short MA is above current long MA, AND previous short MA was below or equal to previous long MA
        if (shortMA_current > longMA_current && shortMA_previous <= longMA_previous) {
            return Signal.BUY;
        }
        // Crossover downwards: current short MA is below current long MA, AND previous short MA was above or equal to previous long MA
        else if (shortMA_current < longMA_current && shortMA_previous >= longMA_previous) {
            return Signal.SELL;
        }
        // Otherwise, no crossover or conditions not met
        else {
            return Signal.HOLD;
        }
    }
}
