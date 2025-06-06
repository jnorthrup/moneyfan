package com.vsiwest.moneyfan.coinbase;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.vsiwest.moneyfan.config.CoinbaseApiConfig;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.apache.commons.codec.binary.Hex;

import javax.crypto.Mac;
import javax.crypto.spec.SecretKeySpec;
import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.security.InvalidKeyException;
import java.security.NoSuchAlgorithmException;
import java.time.Duration;
import java.time.Instant;
import java.util.Collections;
import java.util.List;
import java.util.Map;

public class CoinbaseApiClient {

    private static final Logger logger = LoggerFactory.getLogger(CoinbaseApiClient.class);
    private static final String API_BASE_URL = "https://api.coinbase.com";
    private static final String ACCOUNTS_ENDPOINT = "/api/v3/brokerage/accounts";

    private final CoinbaseApiConfig apiConfig;
    private final HttpClient httpClient; // Use java.net.http.HttpClient
    private final ObjectMapper objectMapper;

    public CoinbaseApiClient(CoinbaseApiConfig apiConfig) {
        this.apiConfig = apiConfig;
        this.httpClient = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(10))
                .build();
        this.objectMapper = new ObjectMapper();
    }

    // Constructor for injecting HttpClient, useful for testing
    public CoinbaseApiClient(CoinbaseApiConfig apiConfig, HttpClient httpClient) { // Use java.net.http.HttpClient
        this.apiConfig = apiConfig;
        this.httpClient = httpClient;
        this.objectMapper = new ObjectMapper();
    }

    // Package-private for testing
    String generateSignature(String timestamp, String method, String requestPath, String body) throws CoinbaseApiException {
        try {
            String prehash = timestamp + method.toUpperCase() + requestPath + (body == null ? "" : body);
            Mac sha256Hmac = Mac.getInstance("HmacSHA256");
            SecretKeySpec secretKey = new SecretKeySpec(apiConfig.getApiSecret().getBytes(StandardCharsets.UTF_8), "HmacSHA256");
            sha256Hmac.init(secretKey);
            byte[] signatureBytes = sha256Hmac.doFinal(prehash.getBytes(StandardCharsets.UTF_8));
            return Hex.encodeHexString(signatureBytes);
        } catch (NoSuchAlgorithmException | InvalidKeyException e) {
            logger.error("Cryptography error during signature generation.", e);
            throw new CoinbaseApiException("Failed to generate API signature due to cryptographic error.", e);
        }
    }

    public List<Map<String, Object>> getAccountBalances() throws CoinbaseApiException {
        String requestPath = ACCOUNTS_ENDPOINT;
        String method = "GET";
        String timestamp = String.valueOf(Instant.now().getEpochSecond());
        String body = ""; // Body is empty for GET request

        String signature;
        try {
            signature = generateSignature(timestamp, method, requestPath, body);
        } catch (CoinbaseApiException e) { // Catch if generateSignature itself throws (already wrapped)
            // Logged in generateSignature, rethrow directly if it's already a CoinbaseApiException
            throw e;
        }


        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(API_BASE_URL + requestPath))
                .header("CB-ACCESS-KEY", apiConfig.getApiKey())
                .header("CB-ACCESS-SIGN", signature)
                .header("CB-ACCESS-TIMESTAMP", timestamp)
                .header("Content-Type", "application/json")
                .GET()
                .build();

        logger.debug("Executing GET request to {} with timestamp {}", request.uri(), timestamp);

        try {
            HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
            int statusCode = response.statusCode();
            String responseBody = response.body();
            logger.debug("Received response with status code: {}. Body: {}", statusCode, responseBody);


            if (statusCode >= 200 && statusCode < 300) {
                if (responseBody == null || responseBody.isEmpty()) {
                    logger.warn("Received empty success response body (status code: {}) from {}", statusCode, request.uri());
                    return Collections.emptyList();
                }
                try {
                    Map<String, Object> responseMap = objectMapper.readValue(responseBody, new TypeReference<Map<String, Object>>() {});
                    if (responseMap.containsKey("accounts") && responseMap.get("accounts") instanceof List) {
                        @SuppressWarnings("unchecked") // Type safety is checked by instanceof
                        List<Map<String, Object>> accounts = (List<Map<String, Object>>) responseMap.get("accounts");
                        logger.info("Successfully fetched {} account(s) from Coinbase.", accounts.size());
                        return accounts;
                    } else {
                        logger.error("Unexpected JSON structure in successful response from {}. Body: {}", request.uri(), responseBody);
                        throw new CoinbaseApiException("Unexpected JSON structure from Coinbase API: 'accounts' array not found.");
                    }
                } catch (IOException e) {
                    logger.error("Failed to parse successful JSON response from {}. Body: {}", request.uri(), responseBody, e);
                    throw new CoinbaseApiException("Failed to parse JSON response from Coinbase API.", e);
                }
            } else {
                logger.error("Coinbase API request to {} failed with status code: {}. Response: {}", request.uri(), statusCode, responseBody);
                throw new CoinbaseApiException("Coinbase API request failed: " + statusCode + " - " + responseBody);
            }
        } catch (InterruptedException e) {
            logger.warn("Coinbase API call to {} interrupted.", request.uri(), e);
            Thread.currentThread().interrupt();
            throw new CoinbaseApiException("Coinbase API call interrupted: " + e.getMessage(), e);
        } catch (IOException e) { // Catch other IOExceptions from httpClient.send
            logger.error("IOException during Coinbase API call to {}.", request.uri(), e);
            throw new CoinbaseApiException("Network or IO error calling Coinbase API: " + e.getMessage(), e);
        }
    }
}
