package com.vsiwest.moneyfan.model;

import org.junit.jupiter.api.Test;
import java.math.BigDecimal;
import java.time.Instant;
import java.util.HashMap;
import java.util.Map;
import static org.junit.jupiter.api.Assertions.*;

class CryptoAssetTest {

    @Test
    void testCryptoAssetCreation() {
        CryptoAsset asset = new CryptoAsset("uuid1", "Bitcoin Wallet", "BTC", new BigDecimal("0.5"), "BTC", Instant.now(), "Coinbase");
        asset.setCurrentPrice(new BigDecimal("30000"));
        asset.setPriceCurrency("USD");

        assertEquals("uuid1", asset.getAssetId());
        assertEquals("BTC", asset.getAssetName()); // Prefers symbol if name is generic
        assertEquals("BTC", asset.getAssetSymbol());
        assertEquals(new BigDecimal("0.5"), asset.getAvailableBalance());
        assertEquals("BTC", asset.getCurrencyOfBalance());
        assertEquals(new BigDecimal("30000"), asset.getCurrentPrice());
        assertEquals("USD", asset.getPriceCurrency());
        assertEquals(AssetType.CRYPTOCURRENCY, asset.getAssetType());
    }

    @Test
    void testFromCoinbaseAccountMap() {
        Map<String, Object> accountMap = new HashMap<>();
        accountMap.put("uuid", "uuid2");
        accountMap.put("name", "Ethereum Account");
        accountMap.put("currency", "ETH");
        Map<String, String> balanceMap = new HashMap<>();
        balanceMap.put("value", "1.234");
        balanceMap.put("currency", "ETH");
        accountMap.put("available_balance", balanceMap);

        CryptoAsset asset = CryptoAsset.fromCoinbaseAccountMap(accountMap, "CoinbaseTest");

        assertEquals("uuid2", asset.getAssetId());
        assertEquals("Ethereum Account", asset.getAssetName());
        assertEquals("ETH", asset.getAssetSymbol());
        assertEquals(new BigDecimal("1.234"), asset.getAvailableBalance());
        assertEquals("ETH", asset.getCurrencyOfBalance());
        assertEquals("CoinbaseTest", asset.getDataSource());
        assertEquals(AssetType.CRYPTOCURRENCY, asset.getAssetType());
        assertNull(asset.getCurrentPrice()); // Price is not set by this factory method
    }
     @Test
    void testCryptoAssetSpecificName() {
        CryptoAsset asset = new CryptoAsset("uuid3", "My Special BTC Investment", "BTC", new BigDecimal("0.1"), "BTC", Instant.now(), "Kraken");
        assertEquals("My Special BTC Investment", asset.getAssetName());
    }
}
