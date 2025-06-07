package com.vsiwest.moneyfan.model;

import java.math.BigDecimal;
import java.time.Instant;

public class LlmToken implements AssetData { // Added implements AssetData
    private String id;
    private String name;
    private String symbol;
    private BigDecimal price;
    private BigDecimal volume24h;
    private String sourceApi;
    private Instant lastUpdated;
    private String priceCurrency; // Added for AssetData

    // Constructors
    public LlmToken() {
        this.priceCurrency = "USD"; // Default
    }

    public LlmToken(String id, String name, String symbol, BigDecimal price, String priceCurrency, BigDecimal volume24h, String sourceApi, Instant lastUpdated) {
        this.id = id;
        this.name = name;
        this.symbol = symbol;
        this.price = price;
        this.priceCurrency = (priceCurrency != null && !priceCurrency.isEmpty()) ? priceCurrency : "USD";
        this.volume24h = volume24h;
        this.sourceApi = sourceApi;
        this.lastUpdated = lastUpdated;
    }

    // Overload constructor for backward compatibility or if price is always USD
    public LlmToken(String id, String name, String symbol, BigDecimal price, BigDecimal volume24h, String sourceApi, Instant lastUpdated) {
        this(id, name, symbol, price, "USD", volume24h, sourceApi, lastUpdated);
    }


    // Getters and Setters (existing)
    public String getId() { return id; }
    public void setId(String id) { this.id = id; }
    public String getName() { return name; }
    public void setName(String name) { this.name = name; }
    public String getSymbol() { return symbol; }
    public void setSymbol(String symbol) { this.symbol = symbol; }
    public BigDecimal getPrice() { return price; }
    public void setPrice(BigDecimal price) { this.price = price; }
    public BigDecimal getVolume24h() { return volume24h; }
    public void setVolume24h(BigDecimal volume24h) { this.volume24h = volume24h; }
    public String getSourceApi() { return sourceApi; } // Will be used for getDataSource
    public void setSourceApi(String sourceApi) { this.sourceApi = sourceApi; }
    // public Instant getLastUpdated() { return lastUpdated; } // Already matches AssetData
    public void setLastUpdated(Instant lastUpdated) { this.lastUpdated = lastUpdated; }

    // AssetData implementation
    @Override
    public String getAssetId() { return this.id; }
    @Override
    public String getAssetName() { return this.name; }
    @Override
    public String getAssetSymbol() { return this.symbol; }
    @Override
    public BigDecimal getCurrentPrice() { return this.price; }
    @Override
    public String getPriceCurrency() { return this.priceCurrency; }
    public void setPriceCurrency(String priceCurrency) { this.priceCurrency = priceCurrency; }
    @Override
    public Instant getLastUpdated() { return this.lastUpdated; }
    @Override
    public String getDataSource() { return this.sourceApi; }
    @Override
    public AssetType getAssetType() { return AssetType.LLM_TOKEN; }

    @Override
    public String toString() { // Consider adding priceCurrency to toString
        return "LlmToken{" +
                "id='" + id + '\'' +
                ", name='" + name + '\'' +
                ", symbol='" + symbol + '\'' +
                ", price=" + price +
                ", priceCurrency='" + priceCurrency + '\'' +
                ", volume24h=" + volume24h +
                ", sourceApi='" + sourceApi + '\'' +
                ", lastUpdated=" + lastUpdated +
                '}';
    }
}
