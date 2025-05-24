package com.example.bikeshed.trading;

import com.example.bikeshed.dsel.Cursor;
import com.example.bikeshed.dsel.D;
import com.example.bikeshed.dsel.Join;
import com.example.bikeshed.dsel.RowVec;
import com.example.bikeshed.dsel.Series;
import com.example.bikeshed.types.ColumnMeta;
import com.example.bikeshed.types.IOMemento;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.function.Function;
import java.util.function.IntFunction;
import java.util.stream.Collectors;

/**
 * Builds `AgentObservation` (a `Cursor`) for the trading agent based on the current market tick
 * and historical data provided by `MarketHistoryProvider`.
 * This demonstrates how DSEL primitives (`Series`, `Cursor`, `Join`) are used to construct
 * the observation space.
 */
public class ObservationBuilder {

    // Define the structure of the observation Cursor
    private final List<ColumnMeta> observationSchema;
    private final int lookbackPeriod; // How many past ticks to include in observation

    /**
     * Constructor for ObservationBuilder.
     *
     * @param lookbackPeriod The number of historical ticks to include in each observation.
     * @param observedAssets The list of asset keys (e.g., "BTC", "ETH") for which data should be observed.
     */
    public ObservationBuilder(int lookbackPeriod, List<String> observedAssets) {
        this.lookbackPeriod = lookbackPeriod;
        this.observationSchema = defineObservationSchema(observedAssets);
    }

    /**
     * Defines the schema for the observation Cursor.
     * Example schema:
     * - Timestamp
     * - For each asset: Open, High, Low, Close, Volume (current tick)
     * - For each asset: Moving Average (e.g., 5-period MA)
     * - For each asset: Relative Strength Index (e.g., 14-period RSI)
     * ... (more indicators)
     *
     * @param assets The list of assets to include in the observation.
     * @return A list of ColumnMeta defining the observation structure.
     */
    private List<ColumnMeta> defineObservationSchema(List<String> assets) {
        List<ColumnMeta> schema = new ArrayList<>();
        schema.add(ColumnMeta.of("timestamp", IOMemento.IO_LONG)); // Global timestamp

        for (String asset : assets) {
            // Current Tick Data
            schema.add(ColumnMeta.of(asset + "_Open", IOMemento.IO_DOUBLE));
            schema.add(ColumnMeta.of(asset + "_High", IOMemento.IO_DOUBLE));
            schema.add(ColumnMeta.of(asset + "_Low", IOMemento.IO_DOUBLE));
            schema.add(ColumnMeta.of(asset + "_Close", IOMemento.IO_DOUBLE));
            schema.add(ColumnMeta.of(asset + "_Volume", IOMemento.IO_DOUBLE));

            // Example: Simple Moving Average (SMA) over lookback period
            schema.add(ColumnMeta.of(asset + "_SMA" + lookbackPeriod, IOMemento.IO_DOUBLE));
            // Example: Volatility (standard deviation of returns)
            schema.add(ColumnMeta.of(asset + "_Volatility", IOMemento.IO_DOUBLE));
        }
        return schema;
    }

    /**
     * Builds a complete `AgentObservation` (a `Cursor`) for the agent.
     * This Cursor contains the current market tick and derived historical indicators.
     *
     * @param currentTick        The most recent market tick.
     * @param historyProvider    Provides access to historical market data.
     * @return A DSEL `Cursor` representing the agent's observation space.
     */
    public Cursor buildObservation(MarketTick currentTick, MarketHistoryProvider historyProvider) {
        // The observation will be a single row Cursor for the current time step,
        // but can be extended to include multiple rows for sequence-based observations.
        // For simplicity, let's create a single-row Cursor observation.

        // Each column in the observation is a Join<Value, () -> ColumnMeta>
        IntFunction<Join<Object, Function<Void, ColumnMeta>>> rowVecProvider = colIndex -> {
            ColumnMeta colMeta = observationSchema.get(colIndex);
            Object value = null;

            // Extract values for current tick
            if (colMeta.getName().equals("timestamp")) {
                value = currentTick.timestamp();
            } else if (colMeta.getName().endsWith("_Open")) {
                String asset = colMeta.getName().replace("_Open", "");
                value = currentTick.data().getOrDefault(asset, new TickData(0.0,0.0,0.0,0.0,0.0)).open();
            } else if (colMeta.getName().endsWith("_High")) {
                String asset = colMeta.getName().replace("_High", "");
                value = currentTick.data().getOrDefault(asset, new TickData(0.0,0.0,0.0,0.0,0.0)).high();
            } else if (colMeta.getName().endsWith("_Low")) {
                String asset = colMeta.getName().replace("_Low", "");
                value = currentTick.data().getOrDefault(asset, new TickData(0.0,0.0,0.0,0.0,0.0)).low();
            } else if (colMeta.getName().endsWith("_Close")) {
                String asset = colMeta.getName().replace("_Close", "");
                value = currentTick.data().getOrDefault(asset, new TickData(0.0,0.0,0.0,0.0,0.0)).close();
            } else if (colMeta.getName().endsWith("_Volume")) {
                String asset = colMeta.getName().replace("_Volume", "");
                value = currentTick.data().getOrDefault(asset, new TickData(0.0,0.0,0.0,0.0,0.0)).volume();
            }
            // Calculate indicators based on historical data
            else if (colMeta.getName().startsWith("BTC_SMA")) { // Example for BTC SMA
                Series<TickData> btcHistory = historyProvider.getHistory("BTC", lookbackPeriod);
                value = calculateSMA(btcHistory, TickData::close);
            } else if (colMeta.getName().startsWith("ETH_SMA")) { // Example for ETH SMA
                Series<TickData> ethHistory = historyProvider.getHistory("ETH", lookbackPeriod);
                value = calculateSMA(ethHistory, TickData::close);
            }
            // Add more indicator calculations here (RSI, Bollinger Bands, MACD, etc.)

            return D.jn(value, (Function<Void, ColumnMeta>) unused -> colMeta);
        };

        // Create a single-row RowVec from the provider
        RowVec observationRow = RowVec.of(observationSchema.size(), rowVecProvider);

        // Return a Cursor containing this single RowVec
        return Cursor.of(1, i -> observationRow);
    }

    /**
     * Calculates Simple Moving Average.
     * @param history Series of TickData.
     * @param valueExtractor Function to extract the price (e.g., TickData::close).
     * @return SMA value.
     */
    private double calculateSMA(Series<TickData> history, Function<TickData, Double> valueExtractor) {
        if (history.isEmpty()) return 0.0;
        return D.reduce(history.map(valueExtractor), 0.0, Double::sum) / history.size();
    }

    // You could add other indicator calculation methods here, e.g.:
    // private double calculateRSI(Series<TickData> history, Function<TickData, Double> valueExtractor) { ... }
    // private double calculateVolatility(Series<TickData> history, Function<TickData, Double> valueExtractor) { ... }
}
