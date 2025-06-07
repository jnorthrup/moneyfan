package com.vsiwest.moneyfan.model;

import org.junit.jupiter.api.Test;
import java.math.BigDecimal;
import java.time.Instant;
import static org.junit.jupiter.api.Assertions.*;

class ToolTest {
    @Test
    void testToolCreationAndGetters() {
        String id = "tool1";
        String name = "Awesome Tool";
        String category = "Productivity";
        BigDecimal price = new BigDecimal("49.99");
        String priceCurrency = "USD";
        String billingCycle = "monthly";
        String sourceApi = "ToolSource";
        Instant now = Instant.now();

        Tool tool = new Tool(id, name, category, price, priceCurrency, billingCycle, sourceApi, now);

        assertEquals(id, tool.getAssetId());
        assertEquals(name, tool.getAssetName());
        assertEquals(name, tool.getAssetSymbol()); // Symbol defaults to name for Tool
        assertEquals(category, tool.getCategory());
        assertEquals(price, tool.getCurrentPrice());
        assertEquals(priceCurrency, tool.getPriceCurrency());
        assertEquals(billingCycle, tool.getBillingCycle());
        assertEquals(sourceApi, tool.getDataSource());
        assertEquals(now, tool.getLastUpdated());
        assertEquals(AssetType.SOFTWARE_TOOL, tool.getAssetType());
    }
}
