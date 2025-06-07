package com.vsiwest.moneyfan.model;

import java.math.BigDecimal;
import java.time.Instant;

public class Tool implements AssetData { // Added implements AssetData
    private String id;
    private String name;
    private String category;
    private BigDecimal price;
    private String billingCycle;
    private String sourceApi;
    private Instant lastUpdated;
    private String priceCurrency; // Added for AssetData

    // Constructors
    public Tool() {
        this.priceCurrency = "USD"; // Default
    }

    public Tool(String id, String name, String category, BigDecimal price, String priceCurrency, String billingCycle, String sourceApi, Instant lastUpdated) {
        this.id = id;
        this.name = name;
        this.category = category;
        this.price = price;
        this.priceCurrency = (priceCurrency != null && !priceCurrency.isEmpty()) ? priceCurrency : "USD";
        this.billingCycle = billingCycle;
        this.sourceApi = sourceApi;
        this.lastUpdated = lastUpdated;
    }

    // Overload constructor for backward compatibility
    public Tool(String id, String name, String category, BigDecimal price, String billingCycle, String sourceApi, Instant lastUpdated) {
        this(id, name, category, price, "USD", billingCycle, sourceApi, lastUpdated);
    }


    // Getters and Setters (existing)
    public String getId() { return id; }
    public void setId(String id) { this.id = id; }
    public String getName() { return name; }
    public void setName(String name) { this.name = name; }
    public String getCategory() { return category; }
    public void setCategory(String category) { this.category = category; }
    public BigDecimal getPrice() { return price; }
    public void setPrice(BigDecimal price) { this.price = price; }
    public String getBillingCycle() { return billingCycle; }
    public void setBillingCycle(String billingCycle) { this.billingCycle = billingCycle; }
    public String getSourceApi() { return sourceApi; }
    public void setSourceApi(String sourceApi) { this.sourceApi = sourceApi; }
    // public Instant getLastUpdated() { return lastUpdated; } // Already matches
    public void setLastUpdated(Instant lastUpdated) { this.lastUpdated = lastUpdated; }

    // AssetData implementation
    @Override
    public String getAssetId() { return this.id; }
    @Override
    public String getAssetName() { return this.name; }
    @Override
    public String getAssetSymbol() { return this.name; } // Tools might not have a distinct "symbol"
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
    public AssetType getAssetType() { return AssetType.SOFTWARE_TOOL; }

    @Override
    public String toString() { // Consider adding priceCurrency
        return "Tool{" +
                "id='" + id + '\'' +
                ", name='" + name + '\'' +
                ", category='" + category + '\'' +
                ", price=" + price +
                ", priceCurrency='" + priceCurrency + '\'' +
                ", billingCycle='" + billingCycle + '\'' +
                ", sourceApi='" + sourceApi + '\'' +
                ", lastUpdated=" + lastUpdated +
                '}';
    }
}
