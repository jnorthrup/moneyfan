package com.vsiwest.moneyfan.model;

import java.math.BigDecimal;
import java.time.Instant;

public class Integration implements AssetData { // Added implements AssetData
    private String id;
    private String name;
    private String type;
    private BigDecimal setupFee;    // This is specific, AssetData.getCurrentPrice might represent monthlyFee
    private BigDecimal monthlyFee;  // This seems like the best fit for AssetData.getCurrentPrice
    private String sourceApi;
    private Instant lastUpdated;
    private String priceCurrency; // Added for AssetData

    // Constructors
    public Integration() {
        this.priceCurrency = "USD"; // Default
    }

    public Integration(String id, String name, String type, BigDecimal setupFee, BigDecimal monthlyFee, String priceCurrency, String sourceApi, Instant lastUpdated) {
        this.id = id;
        this.name = name;
        this.type = type;
        this.setupFee = setupFee;
        this.monthlyFee = monthlyFee;
        this.priceCurrency = (priceCurrency != null && !priceCurrency.isEmpty()) ? priceCurrency : "USD";
        this.sourceApi = sourceApi;
        this.lastUpdated = lastUpdated;
    }

    // Overload for backward compatibility
    public Integration(String id, String name, String type, BigDecimal setupFee, BigDecimal monthlyFee, String sourceApi, Instant lastUpdated) {
        this(id, name, type, setupFee, monthlyFee, "USD", sourceApi, lastUpdated);
    }

    // Getters and Setters (existing)
    public String getId() { return id; }
    public void setId(String id) { this.id = id; }
    public String getName() { return name; }
    public void setName(String name) { this.name = name; }
    public String getType() { return type; }
    public void setType(String type) { this.type = type; }
    public BigDecimal getSetupFee() { return setupFee; }
    public void setSetupFee(BigDecimal setupFee) { this.setupFee = setupFee; }
    public BigDecimal getMonthlyFee() { return monthlyFee; }
    public void setMonthlyFee(BigDecimal monthlyFee) { this.monthlyFee = monthlyFee; }
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
    public String getAssetSymbol() { return this.name; } // Integrations might not have a "symbol"
    @Override
    public BigDecimal getCurrentPrice() { return this.monthlyFee; } // Using monthlyFee as the "current price"
    @Override
    public String getPriceCurrency() { return this.priceCurrency; }
    public void setPriceCurrency(String priceCurrency) { this.priceCurrency = priceCurrency; }
    @Override
    public Instant getLastUpdated() { return this.lastUpdated; }
    @Override
    public String getDataSource() { return this.sourceApi; }
    @Override
    public AssetType getAssetType() { return AssetType.INTEGRATION_SERVICE; }

    @Override
    public String toString() { // Consider adding priceCurrency
        return "Integration{" +
                "id='" + id + '\'' +
                ", name='" + name + '\'' +
                ", type='" + type + '\'' +
                ", setupFee=" + setupFee +
                ", monthlyFee=" + monthlyFee +
                ", priceCurrency='" + priceCurrency + '\'' +
                ", sourceApi='" + sourceApi + '\'' +
                ", lastUpdated=" + lastUpdated +
                '}';
    }
}
