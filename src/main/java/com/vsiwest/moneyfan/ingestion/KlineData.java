package com.vsiwest.moneyfan.ingestion;

import java.util.Objects;

public class KlineData {
    private long openTime;
    private double openPrice;
    private double highPrice;
    private double lowPrice;
    private double closePrice;
    private double volume;
    private long closeTime;
    private double quoteAssetVolume;
    private int numberOfTrades;
    private double takerBuyBaseAssetVolume;
    private double takerBuyQuoteAssetVolume;

    public KlineData(long openTime, double openPrice, double highPrice, double lowPrice, double closePrice,
                     double volume, long closeTime, double quoteAssetVolume, int numberOfTrades,
                     double takerBuyBaseAssetVolume, double takerBuyQuoteAssetVolume) {
        this.openTime = openTime;
        this.openPrice = openPrice;
        this.highPrice = highPrice;
        this.lowPrice = lowPrice;
        this.closePrice = closePrice;
        this.volume = volume;
        this.closeTime = closeTime;
        this.quoteAssetVolume = quoteAssetVolume;
        this.numberOfTrades = numberOfTrades;
        this.takerBuyBaseAssetVolume = takerBuyBaseAssetVolume;
        this.takerBuyQuoteAssetVolume = takerBuyQuoteAssetVolume;
    }

    public long getOpenTime() {
        return openTime;
    }

    public double getOpenPrice() {
        return openPrice;
    }

    public double getHighPrice() {
        return highPrice;
    }

    public double getLowPrice() {
        return lowPrice;
    }

    public double getClosePrice() {
        return closePrice;
    }

    public double getVolume() {
        return volume;
    }

    public long getCloseTime() {
        return closeTime;
    }

    public double getQuoteAssetVolume() {
        return quoteAssetVolume;
    }

    public int getNumberOfTrades() {
        return numberOfTrades;
    }

    public double getTakerBuyBaseAssetVolume() {
        return takerBuyBaseAssetVolume;
    }

    public double getTakerBuyQuoteAssetVolume() {
        return takerBuyQuoteAssetVolume;
    }

    @Override
    public String toString() {
        return "KlineData{" +
                "openTime=" + openTime +
                ", openPrice=" + openPrice +
                ", highPrice=" + highPrice +
                ", lowPrice=" + lowPrice +
                ", closePrice=" + closePrice +
                ", volume=" + volume +
                ", closeTime=" + closeTime +
                ", quoteAssetVolume=" + quoteAssetVolume +
                ", numberOfTrades=" + numberOfTrades +
                ", takerBuyBaseAssetVolume=" + takerBuyBaseAssetVolume +
                ", takerBuyQuoteAssetVolume=" + takerBuyQuoteAssetVolume +
                '}';
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        KlineData klineData = (KlineData) o;
        return openTime == klineData.openTime &&
                Double.compare(klineData.openPrice, openPrice) == 0 &&
                Double.compare(klineData.highPrice, highPrice) == 0 &&
                Double.compare(klineData.lowPrice, lowPrice) == 0 &&
                Double.compare(klineData.closePrice, closePrice) == 0 &&
                Double.compare(klineData.volume, volume) == 0 &&
                closeTime == klineData.closeTime &&
                Double.compare(klineData.quoteAssetVolume, quoteAssetVolume) == 0 &&
                numberOfTrades == klineData.numberOfTrades &&
                Double.compare(klineData.takerBuyBaseAssetVolume, takerBuyBaseAssetVolume) == 0 &&
                Double.compare(klineData.takerBuyQuoteAssetVolume, takerBuyQuoteAssetVolume) == 0;
    }

    @Override
    public int hashCode() {
        return Objects.hash(openTime, openPrice, highPrice, lowPrice, closePrice, volume, closeTime,
                            quoteAssetVolume, numberOfTrades, takerBuyBaseAssetVolume, takerBuyQuoteAssetVolume);
    }
}
