package com.vsiwest.moneyfan.backtest;

import com.vsiwest.moneyfan.strategy.Signal;
import java.util.Objects;

public class Trade {
    private final Signal signal; // BUY or SELL
    private final double entryPrice;
    private final long entryTime;
    private final double size;

    private Double exitPrice; // Can be null if position is still open
    private Long exitTime;    // Can be null if position is still open
    private Double profitOrLoss; // Can be null if position is still open

    public Trade(Signal signal, double entryPrice, long entryTime, double size) {
        if (signal != Signal.BUY && signal != Signal.SELL) {
            throw new IllegalArgumentException("Trade signal must be BUY or SELL.");
        }
        this.signal = signal;
        this.entryPrice = entryPrice;
        this.entryTime = entryTime;
        this.size = size;
    }

    public void close(double exitPrice, long exitTime) {
        this.exitPrice = exitPrice;
        this.exitTime = exitTime;
        calculateProfitOrLoss();
    }

    private void calculateProfitOrLoss() {
        if (this.exitPrice == null) {
            this.profitOrLoss = null; // Position not closed
            return;
        }
        if (signal == Signal.BUY) {
            this.profitOrLoss = (this.exitPrice - this.entryPrice) * this.size;
        } else { // Signal.SELL
            this.profitOrLoss = (this.entryPrice - this.exitPrice) * this.size;
        }
    }

    public Signal getSignal() {
        return signal;
    }

    public double getEntryPrice() {
        return entryPrice;
    }

    public Double getExitPrice() {
        return exitPrice;
    }

    public long getEntryTime() {
        return entryTime;
    }

    public Long getExitTime() {
        return exitTime;
    }

    public double getSize() {
        return size;
    }

    public Double getProfitOrLoss() {
        return profitOrLoss;
    }

    public boolean isOpen() {
        return this.exitPrice == null;
    }

    @Override
    public String toString() {
        return "Trade{" +
                "signal=" + signal +
                ", entryPrice=" + entryPrice +
                ", exitPrice=" + (exitPrice != null ? exitPrice : "N/A") +
                ", entryTime=" + entryTime +
                ", exitTime=" + (exitTime != null ? exitTime : "N/A") +
                ", size=" + size +
                ", profitOrLoss=" + (profitOrLoss != null ? String.format("%.2f", profitOrLoss) : "N/A") +
                '}';
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Trade trade = (Trade) o;
        return Double.compare(trade.entryPrice, entryPrice) == 0 &&
                entryTime == trade.entryTime &&
                Double.compare(trade.size, size) == 0 &&
                signal == trade.signal &&
                Objects.equals(exitPrice, trade.exitPrice) &&
                Objects.equals(exitTime, trade.exitTime) &&
                Objects.equals(profitOrLoss, trade.profitOrLoss);
    }

    @Override
    public int hashCode() {
        return Objects.hash(signal, entryPrice, exitPrice, entryTime, exitTime, size, profitOrLoss);
    }
}
