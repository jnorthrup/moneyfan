package com.vsiwest.bikeshed.trading;

import com.example.bikeshed.dsel.Join;
import com.example.bikeshed.dsel.D;
import com.example.bikeshed.dsel.RowVec;
import com.example.bikeshed.dsel.Series;
import com.example.bikeshed.types.ColumnMeta;
import com.example.bikeshed.types.IOMemento;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Function;

/**
 * Simulates trade execution and manages agent wallet state.
 * Emphasizes that DSEL's immutable `Join`-based records reduce the need for complex locking
 * on data *access*. Mutations (e.g., agent actions, wallet state updates) occur on
 * agent-local or transaction-specific mutable state, which then gets integrated compositionally.
 */
public class ExecutionEngine {

    private AtomicReference<WalletState> currentWalletState;
    private PriceOracle priceOracle; // Dependency for getting current asset prices

    /**
     * Factory method for `FillInfo`.
     */
    public static FillInfo createFillInfo(String assetKey, String side, double quantity, double price, long timestamp) {
        return new FillInfo(assetKey, side, quantity, price, timestamp);
    }

    /**
     * Factory method for `ExecutionResult`.
     */
    public static ExecutionResult createExecutionResult(List<FillInfo> fills, double pnlChange, WalletState newWalletState) {
        return new ExecutionResult(fills, pnlChange, newWalletState);
    }


    public ExecutionEngine(PriceOracle priceOracle) {
        this.priceOracle = Objects.requireNonNull(priceOracle);
    }

    /**
     * Initializes the execution engine with an initial wallet state.
     * @param initialWalletState The starting wallet state.
     */
    public void initialize(WalletState initialWalletState) {
        this.currentWalletState = new AtomicReference<>(initialWalletState);
    }

    /**
     * Processes a single market tick and an agent's action, producing an execution result.
     * This method is designed to be called concurrently, with internal state managed safely.
     *
     * @param currentTick The current market tick data.
     * @param agentAction The actions proposed by the agent (e.g., `double[]` for asset allocations).
     * @return An `ExecutionResult` representing the outcome.
     */
    public ExecutionResult processTick(MarketTick currentTick, double[] agentAction) {
        // Here, we simulate trade execution.
        // `agentAction` is a `double[]` representing proposed actions.
        // Let's assume `agentAction[0]` is for BTC, `agentAction[1]` for ETH, etc.
        // A positive value could mean BUY, negative means SELL.
        // The magnitude could be the amount in base currency (e.g., USD).

        WalletState oldWallet = currentWalletState.get();
        Map<String, Double> newBalances = new HashMap<>(oldWallet.balances());
        List<FillInfo> fills = new ArrayList<>();
        double pnlChange = 0.0; // Profit and Loss change for this tick

        // Simplified: Process action for a single asset (e.g., "BTC")
        // In a real simulator, agentAction would be more structured,
        // mapping to specific assets and order types.
        // Assuming agentAction[0] controls "BTC" amount (in USD terms)
        // and agentAction[1] controls "USD" amount.

        // Get the current price of BTC/USD from the oracle using the currentTick context
        double btcUsdPrice = priceOracle.getReferencePrice("BTC_USD", currentTick); // This implies a specific asset key

        // Example: Assume agentAction[0] is desired BTC exposure (positive for buy, negative for sell)
        // This is a highly simplified model. A real one would have order books, slippage, etc.
        double desiredBtcChangeUsd = agentAction.length > 0 ? agentAction[0] : 0.0;

        double currentBtcBalance = newBalances.getOrDefault("BTC", 0.0);
        double currentUsdBalance = newBalances.getOrDefault("USD", 0.0);

        if (desiredBtcChangeUsd > 0) { // BUY BTC with USD
            double btcToBuy = desiredBtcChangeUsd / btcUsdPrice;
            if (currentUsdBalance >= desiredBtcChangeUsd) {
                newBalances.put("USD", currentUsdBalance - desiredBtcChangeUsd);
                newBalances.put("BTC", currentBtcBalance + btcToBuy);
                fills.add(createFillInfo("BTC_USD", "BUY", btcToBuy, btcUsdPrice, currentTick.timestamp()));
            } else {
                System.out.println("Insufficient USD to buy BTC. Available: " + currentUsdBalance);
                // Can partial fill here, or just skip. For simplicity, skip.
            }
        } else if (desiredBtcChangeUsd < 0) { // SELL BTC for USD
            double btcToSell = -desiredBtcChangeUsd / btcUsdPrice; // Convert USD value to BTC amount
            if (currentBtcBalance >= btcToSell) {
                newBalances.put("BTC", currentBtcBalance - btcToSell);
                newBalances.put("USD", currentUsdBalance + (-desiredBtcChangeUsd));
                fills.add(createFillInfo("BTC_USD", "SELL", btcToSell, btcUsdPrice, currentTick.timestamp()));
            } else {
                System.out.println("Insufficient BTC to sell. Available: " + currentBtcBalance);
            }
        }

        // Calculate PnL change (simplistic: based on paper PnL from previous tick to current tick)
        // A more robust PnL calculation would involve marked-to-market valuations of all assets.
        // This needs a "previous tick" context, which is outside this `processTick` scope.
        // For now, let's just make PnL change 0 or related to trade execution fees/slippage.
        // In a true simulator, PnL would be (current_portfolio_value - previous_portfolio_value - funds_in/out).
        // Let's assume PnL is determined by an external reward system or a simple fee deduction.
        // For now, pnlChange represents commission or fixed fee.
        pnlChange = - (fills.size() * 0.01); // Example: 0.01 USD fee per trade

        WalletState newWallet = new WalletState(newBalances);
        currentWalletState.set(newWallet); // Atomically update shared state.

        return createExecutionResult(fills, pnlChange, newWallet);
    }
}

/** Represents a trade fill. */
class FillInfo extends Join<String, Join<String, Join<Double, Join<Double, Long>>>> {
    FillInfo(String assetKey, String side, double quantity, double price, long timestamp) {
        super(assetKey, D.jn(side, D.jn(quantity, D.jn(price, timestamp))));
    }

    public String assetKey() { return a(); }
    public String side() { return b().a(); }
    public double quantity() { return b().b().a(); }
    public double price() { return b().b().b().a(); }
    public long timestamp() { return b().b().b().b(); }
}

/** Represents the result of an execution step. */
class ExecutionResult extends Join<List<FillInfo>, Join<Double, WalletState>> {
    ExecutionResult(List<FillInfo> fills, double pnlChange, WalletState newWalletState) {
        super(Collections.unmodifiableList(fills), D.jn(pnlChange, newWalletState));
    }

    public List<FillInfo> fills() { return a(); }
    public double pnlChange() { return b().a(); }
    public WalletState newWalletState() { return b().b(); }
}

// Dummy PriceOracle for simulation. In real life, it would fetch actual prices.
interface PriceOracle {
    double getReferencePrice(String asset, MarketTick currentTick);
}

class SimplePriceOracle implements PriceOracle {
    @Override
    public double getReferencePrice(String asset, MarketTick currentTick) {
        // In a real scenario, this would look up the actual close price for the asset
        // For simplicity, assume asset is "BTC_USD" and we get its close price.
        TickData data = currentTick.data().get(asset.split("_")[0]); // Crude way to get asset data
        return (data != null) ? data.close() : 0.0; // Return close price or 0.0 if not found
    }
}
