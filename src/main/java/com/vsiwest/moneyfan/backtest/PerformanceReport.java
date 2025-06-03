package com.vsiwest.moneyfan.backtest;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public class PerformanceReport {

    private int totalTrades;
    private int winningTrades;
    private int losingTrades;
    private double totalProfitLoss;
    private double profitFactor; // Gross Profit / Gross Loss
    private double maxDrawdown; // Placeholder
    private double averageWin;
    private double averageLoss;
    private double sharpeRatio; // Placeholder
    private List<Trade> trades;

    public PerformanceReport(List<Trade> trades) {
        this.trades = new ArrayList<>(Objects.requireNonNull(trades, "Trades list cannot be null"));
        calculateMetrics();
    }

    private void calculateMetrics() {
        this.totalTrades = this.trades.size();
        if (this.totalTrades == 0) {
            // Avoid division by zero or issues with no trades
            this.totalProfitLoss = 0;
            this.winningTrades = 0;
            this.losingTrades = 0;
            this.averageWin = 0;
            this.averageLoss = 0;
            this.profitFactor = 0;
            this.maxDrawdown = 0; // Placeholder
            this.sharpeRatio = 0; // Placeholder
            return;
        }

        double grossProfit = 0;
        double grossLoss = 0;
        double sumOfWins = 0;
        double sumOfLosses = 0;

        for (Trade trade : this.trades) {
            if (trade.isOpen() || trade.getProfitOrLoss() == null) {
                // Skip open trades for P&L calculations or if P&L is not set
                this.totalTrades--; // Adjust total trades considered for P&L metrics
                continue;
            }
            double pnl = trade.getProfitOrLoss();
            this.totalProfitLoss += pnl;
            if (pnl > 0) {
                this.winningTrades++;
                sumOfWins += pnl;
                grossProfit += pnl;
            } else if (pnl < 0) {
                this.losingTrades++;
                sumOfLosses += pnl; // pnl is already negative
                grossLoss += Math.abs(pnl); // Gross loss is positive
            }
        }

        this.averageWin = (this.winningTrades > 0) ? sumOfWins / this.winningTrades : 0;
        this.averageLoss = (this.losingTrades > 0) ? sumOfLosses / this.losingTrades : 0; // Will be negative or zero

        if (grossLoss > 0) {
            this.profitFactor = grossProfit / grossLoss;
        } else if (grossProfit > 0) {
            this.profitFactor = Double.POSITIVE_INFINITY; // All wins, no losses
        } else {
            this.profitFactor = 0; // No wins, no losses
        }

        // Placeholders for complex metrics
        this.maxDrawdown = 0.0; // TODO: Implement Max Drawdown calculation
        this.sharpeRatio = 0.0; // TODO: Implement Sharpe Ratio calculation
    }

    // Getter methods
    public int getTotalTrades() {
        return totalTrades;
    }

    public int getWinningTrades() {
        return winningTrades;
    }

    public int getLosingTrades() {
        return losingTrades;
    }

    public double getTotalProfitLoss() {
        return totalProfitLoss;
    }

    public double getProfitFactor() {
        return profitFactor;
    }

    public double getMaxDrawdown() {
        return maxDrawdown;
    }

    public double getAverageWin() {
        return averageWin;
    }

    public double getAverageLoss() {
        return averageLoss;
    }

    public double getSharpeRatio() {
        return sharpeRatio;
    }

    public List<Trade> getTrades() {
        return new ArrayList<>(trades); // Return a copy for immutability
    }

    @Override
    public String toString() {
        return "PerformanceReport{" + "\n" +
                "  totalTrades=" + totalTrades + ",\n" +
                "  winningTrades=" + winningTrades + ",\n" +
                "  losingTrades=" + losingTrades + ",\n" +
                "  totalProfitLoss=" + String.format("%.2f", totalProfitLoss) + ",\n" +
                "  profitFactor=" + String.format("%.2f", profitFactor) + ",\n" +
                "  averageWin=" + String.format("%.2f", averageWin) + ",\n" +
                "  averageLoss=" + String.format("%.2f", averageLoss) + ",\n" +
                "  maxDrawdown=" + String.format("%.2f", maxDrawdown) + " (placeholder),\n" +
                "  sharpeRatio=" + String.format("%.2f", sharpeRatio) + " (placeholder)\n" +
                '}';
    }
}
