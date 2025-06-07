package com.vsiwest.moneyfan.model;

import java.math.BigDecimal;
import java.time.Instant;
import java.util.Map;

public class CryptoAsset implements AssetData {
    private String id; // From account "uuid"
    private String name; // From account "name"
    private String symbol; // From account "currency"
    private BigDecimal availableBalance; // From account "available_balance"."value"
    private String currencyOfBalance; // From account "available_balance"."currency"
    private BigDecimal currentPrice; // This would need to be fetched/populated from a price API
    private String priceCurrency;    // e.g., USD
    private Instant lastUpdated;     // Timestamp of when this CryptoAsset object was created/updated
    private String dataSource;       // e.g., "Coinbase API"

    // Constructor, getters, setters

    public CryptoAsset(String id, String name, String symbol, BigDecimal availableBalance, String currencyOfBalance, Instant lastUpdated, String dataSource) {
        this.id = id;
        this.name = name;
        this.symbol = symbol;
        this.availableBalance = availableBalance;
        this.currencyOfBalance = currencyOfBalance;
        this.lastUpdated = lastUpdated;
        this.dataSource = dataSource;
        // Default price currency to USD, price itself can be set later
        this.priceCurrency = "USD";
    }

    // Example: Factory method to create from the Map structure used in CoinbaseApiClient
    public static CryptoAsset fromCoinbaseAccountMap(Map<String, Object> accountMap, String dataSource) {
        String id = (String) accountMap.get("uuid");
        String name = (String) accountMap.get("name");
        String currencyCode = (String) accountMap.get("currency");

        Map<String, String> availableBalanceMap = (Map<String, String>) accountMap.get("available_balance");
        BigDecimal availableBalance = new BigDecimal(availableBalanceMap.get("value"));
        String balanceCurrency = availableBalanceMap.get("currency");

        // Price would typically come from another API call using the currencyCode (symbol)
        // For now, it's not set here. lastUpdated is when this conversion happens.
        return new CryptoAsset(id, name, currencyCode, availableBalance, balanceCurrency, Instant.now(), dataSource);
    }


    @Override
    public String getAssetId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    @Override
    public String getAssetName() {
        // If name is too generic like "BTC Wallet", prefer symbol or a combination
        return (name != null && !name.equalsIgnoreCase(symbol + " Wallet")) ? name : symbol;
    }

    public void setName(String name) {
        this.name = name;
    }

    @Override
    public String getAssetSymbol() {
        return symbol;
    }

    public void setSymbol(String symbol) {
        this.symbol = symbol;
    }

    public BigDecimal getAvailableBalance() {
        return availableBalance;
    }

    public void setAvailableBalance(BigDecimal availableBalance) {
        this.availableBalance = availableBalance;
    }

    public String getCurrencyOfBalance() {
        return currencyOfBalance;
    }

    public void setCurrencyOfBalance(String currencyOfBalance) {
        this.currencyOfBalance = currencyOfBalance;
    }

    @Override
    public BigDecimal getCurrentPrice() {
        return currentPrice;
    }

    public void setCurrentPrice(BigDecimal currentPrice) {
        this.currentPrice = currentPrice;
    }

    @Override
    public String getPriceCurrency() {
        return priceCurrency;
    }

    public void setPriceCurrency(String priceCurrency) {
        this.priceCurrency = priceCurrency;
    }

    @Override
    public Instant getLastUpdated() {
        return lastUpdated;
    }

    public void setLastUpdated(Instant lastUpdated) {
        this.lastUpdated = lastUpdated;
    }

    @Override
    public String getDataSource() {
        return dataSource;
    }

    public void setDataSource(String dataSource) {
        this.dataSource = dataSource;
    }

    @Override
    public AssetType getAssetType() {
        return AssetType.CRYPTOCURRENCY;
    }

    @Override
    public String toString() {
        return "CryptoAsset{" +
                "id='" + id + '\'' +
                ", name='" + name + '\'' +
                ", symbol='" + symbol + '\'' +
                ", availableBalance=" + availableBalance +
                ", currencyOfBalance='" + currencyOfBalance + '\'' +
                ", currentPrice=" + currentPrice +
                ", priceCurrency='" + priceCurrency + '\'' +
                ", lastUpdated=" + lastUpdated +
                ", dataSource='" + dataSource + '\'' +
                '}';
    }
}
