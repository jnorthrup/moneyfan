package com.vsiwest.moneyfan.model;

import org.junit.jupiter.api.Test;
import java.math.BigDecimal;
import java.time.Instant;
import static org.junit.jupiter.api.Assertions.*;

class LlmTokenTest {

    @Test
    void testLlmTokenCreationAndGetters() {
        String id = "token1";
        String name = "Test LLM Token";
        String symbol = "TLT";
        BigDecimal price = new BigDecimal("0.05");
        String priceCurrency = "USD";
        BigDecimal volume24h = new BigDecimal("100000");
        String sourceApi = "TestAPI";
        Instant now = Instant.now();

        LlmToken token = new LlmToken(id, name, symbol, price, priceCurrency, volume24h, sourceApi, now);

        assertEquals(id, token.getAssetId());
        assertEquals(name, token.getAssetName());
        assertEquals(symbol, token.getAssetSymbol());
        assertEquals(price, token.getCurrentPrice());
        assertEquals(priceCurrency, token.getPriceCurrency());
        assertEquals(volume24h, token.getVolume24h());
        assertEquals(sourceApi, token.getDataSource());
        assertEquals(now, token.getLastUpdated());
        assertEquals(AssetType.LLM_TOKEN, token.getAssetType());
    }

    @Test
    void testLlmTokenDefaultPriceCurrency() {
        LlmToken token = new LlmToken("id", "name", "sym", BigDecimal.ONE, BigDecimal.TEN, "api", Instant.now());
        assertEquals("USD", token.getPriceCurrency());
    }
}
