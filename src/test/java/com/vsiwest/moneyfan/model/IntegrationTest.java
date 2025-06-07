package com.vsiwest.moneyfan.model;

import org.junit.jupiter.api.Test;
import java.math.BigDecimal;
import java.time.Instant;
import static org.junit.jupiter.api.Assertions.*;

class IntegrationTest {
    @Test
    void testIntegrationCreationAndGetters() {
        String id = "int1";
        String name = "CRM Connector";
        String type = "Connector";
        BigDecimal setupFee = new BigDecimal("500");
        BigDecimal monthlyFee = new BigDecimal("75");
        String priceCurrency = "USD";
        String sourceApi = "IntegrationSource";
        Instant now = Instant.now();

        Integration integration = new Integration(id, name, type, setupFee, monthlyFee, priceCurrency, sourceApi, now);

        assertEquals(id, integration.getAssetId());
        assertEquals(name, integration.getAssetName());
        assertEquals(name, integration.getAssetSymbol()); // Symbol defaults to name
        assertEquals(type, integration.getType());
        assertEquals(setupFee, integration.getSetupFee());
        assertEquals(monthlyFee, integration.getMonthlyFee());
        assertEquals(monthlyFee, integration.getCurrentPrice()); // CurrentPrice maps to monthlyFee
        assertEquals(priceCurrency, integration.getPriceCurrency());
        assertEquals(sourceApi, integration.getDataSource());
        assertEquals(now, integration.getLastUpdated());
        assertEquals(AssetType.INTEGRATION_SERVICE, integration.getAssetType());
    }
}
