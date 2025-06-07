package com.vsiwest.moneyfan.model;

import java.math.BigDecimal;
import java.time.Instant;

public interface AssetData {
    String getAssetId();         // A unique identifier for the asset instance
    String getAssetName();       // A human-readable name for the asset
    String getAssetSymbol();     // A ticker symbol or short code (can be same as name if no symbol)
    BigDecimal getCurrentPrice(); // Current price in a standard currency (e.g., USD)
    String getPriceCurrency();   // The currency of getCurrentPrice() (e.g., "USD")
    Instant getLastUpdated();    // Timestamp of the last data update
    String getDataSource();      // Source of the data (e.g., API name, website)
    AssetType getAssetType();    // Enum to define the type of asset
}
