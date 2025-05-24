package com.example.bikeshed.trading;

import com.example.bikeshed.dsel.D;
import com.example.bikeshed.dsel.Series;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.function.IntFunction;
import java.util.stream.Collectors;
import java.util.List;

/**
 * Provides historical market data to agents, enabling efficient lookback periods.
 * This class maintains a sliding window of historical `TickData` for each asset.
 *
 * Implements the shared timeline concept using immutable DSEL `Series` backed by
 * efficient, potentially concurrent data structures (like circular buffers/queues).
 */
public class MarketHistoryProvider {

    // Using ConcurrentHashMap to store history for multiple assets
    // Each asset's history is stored in a ConcurrentLinkedQueue for simplicity.
    // For large scale, a more efficient circular buffer (custom or external lib) would be needed.
    private final Map<String, ConcurrentLinkedQueue<TickData>> historyQueues;
    private final int capacityPerAsset; // Max number of ticks to store per asset

    public MarketHistoryProvider(int capacityPerAsset) {
        this.capacityPerAsset = capacityPerAsset;
        this.historyQueues = new ConcurrentHashMap<>();
    }

    /**
     * Adds a new market tick to the history. This updates the shared timeline.
     *
     * @param marketTick The new market tick to add.
     */
    public void addTick(MarketTick marketTick) {
        // Iterate through each asset's data in the tick
        marketTick.data().forEach((assetKey, tickData) -> {
            ConcurrentLinkedQueue<TickData> queue = historyQueues.computeIfAbsent(assetKey, k -> new ConcurrentLinkedQueue<>());
            queue.offer(tickData); // Add new tick

            // Evict oldest if capacity exceeded
            while (queue.size() > capacityPerAsset) {
                queue.poll(); // Remove oldest
            }
        });
    }

    /**
     * Retrieves historical data for a specific asset.
     * Returns a `Series<TickData>` which is an immutable, cursor-based view
     * of the historical data.
     *
     * @param assetKey The key of the asset (e.g., "BTC", "ETH").
     * @param ticks    The number of recent ticks to retrieve (lookback period).
     * @return A `Series<TickData>` representing the historical data.
     */
    public Series<TickData> getHistory(String assetKey, int ticks) {
        ConcurrentLinkedQueue<TickData> queue = historyQueues.get(assetKey);
        if (queue == null || queue.isEmpty()) {
            return D.sr(0, i -> null); // Return empty Series if no history
        }

        // Convert queue to a List to enable indexed access for Series
        List<TickData> historyList = queue.stream().collect(Collectors.toList());

        // Ensure we don't request more ticks than available
        int actualTicks = Math.min(ticks, historyList.size());
        int startIndex = historyList.size() - actualTicks;

        // Return a Series that wraps the relevant portion of the history list.
        // This makes it a cursor-based access.
        return D.sr(actualTicks, i -> historyList.get(startIndex + i));
    }
}
