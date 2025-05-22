package com.moneyfan.integration;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.moneyfan.dsl.core.JPair;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.math.BigDecimal;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;

/**
 * Integration test that fetches recent BTCUSDT 1-minute klines from Binance public API,
 * converts them into domain-specific {@link JPair} instances (openTime, closePrice),
 * and asserts basic invariants. The test is skipped if the Binance endpoint is unreachable.
 */
public class BTCUSDTKlinesTest {

    private static final String ENDPOINT = "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1m&limit=5";

    @Test
    @DisplayName("Fetch BTCUSDT klines and observify into JPair list")
    void testFetchAndObservify() throws Exception {
        HttpClient client = HttpClient.newHttpClient();
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(ENDPOINT))
                .GET()
                .build();

        HttpResponse<String> response;
        try {
            response = client.send(request, HttpResponse.BodyHandlers.ofString());
        } catch (Exception e) {
            // If we cannot reach the endpoint (offline CI, network blocked), skip the test
            Assumptions.abort("Cannot reach Binance API: " + e.getMessage());
            return; // unreachable, but compiler satisfied
        }

        if (response.statusCode() != 200) {
            Assumptions.abort("Unexpected HTTP status: " + response.statusCode());
        }

        ObjectMapper mapper = new ObjectMapper();
        JsonNode root = mapper.readTree(response.body());
        Assertions.assertTrue(root.isArray(), "Response should be JSON array");

        List<JPair<Instant, BigDecimal>> candles = new ArrayList<>();
        for (JsonNode kline : root) {
            long openTimeMillis = kline.get(0).asLong();
            String closePriceStr = kline.get(4).asText();
            Instant openTime = Instant.ofEpochMilli(openTimeMillis);
            BigDecimal closePrice = new BigDecimal(closePriceStr);
            candles.add(JPair.of(openTime, closePrice));
        }

        // Observify: ensure we have exactly the limit requested
        Assertions.assertEquals(5, candles.size(), "Should create 5 candle pairs");
        // Basic sanity: close price should be positive
        candles.forEach(pair -> Assertions.assertTrue(pair.second().compareTo(BigDecimal.ZERO) > 0));
    }
}