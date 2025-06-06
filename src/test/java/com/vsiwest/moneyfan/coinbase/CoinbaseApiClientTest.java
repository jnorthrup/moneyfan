package com.vsiwest.moneyfan.coinbase;

import com.fasterxml.jackson.databind.ObjectMapper;
// Removed duplicate ObjectMapper import
import com.vsiwest.moneyfan.config.CoinbaseApiConfig;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
// Http specific imports from java.net.http
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
// HttpStatus equivalent if needed, or use integer values directly. For this test, integer values are fine.
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
// Removed duplicate ExtendWith, ArgumentCaptor, Mock imports
import org.mockito.junit.jupiter.MockitoExtension;

import java.io.IOException;
import java.net.URI;
import java.time.Instant;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class CoinbaseApiClientTest {

    @Mock
    private CoinbaseApiConfig mockApiConfig;

    @Mock
    private HttpClient mockHttpClient; // java.net.http.HttpClient

    @Mock
    private HttpResponse<String> mockHttpResponse; // java.net.http.HttpResponse

    private CoinbaseApiClient apiClient;
    private final ObjectMapper objectMapper = new ObjectMapper();

    @BeforeEach
    void setUp() {
        lenient().when(mockApiConfig.getApiKey()).thenReturn("testApiKey");
        lenient().when(mockApiConfig.getApiSecret()).thenReturn("testApiSecret");
        apiClient = new CoinbaseApiClient(mockApiConfig, mockHttpClient);
    }

    @Test
    @SuppressWarnings("unchecked") // For mockHttpResponse.body() cast if needed, though direct string is fine
    void getAccountBalances_success() throws CoinbaseApiException, IOException, InterruptedException {
        // Prepare mock response
        Map<String, Object> account1 = Map.of("id", "acc1", "balance", Map.of("amount", "100.00", "currency", "USD"));
        Map<String, Object> account2 = Map.of("id", "acc2", "balance", Map.of("amount", "5.00", "currency", "BTC"));
        Map<String, Object> apiResponseMap = Map.of("accounts", List.of(account1, account2));
        String mockJsonResponse = objectMapper.writeValueAsString(apiResponseMap);

        when(mockHttpResponse.statusCode()).thenReturn(200);
        when(mockHttpResponse.body()).thenReturn(mockJsonResponse);
        when(mockHttpClient.send(any(HttpRequest.class), any(HttpResponse.BodyHandler.class)))
                .thenReturn(mockHttpResponse);

        // Execute
        List<Map<String, Object>> balances = apiClient.getAccountBalances();

        // Verify
        assertNotNull(balances);
        assertEquals(2, balances.size());
        assertEquals("acc1", balances.get(0).get("id"));
        assertEquals("acc2", balances.get(1).get("id"));

        // Verify headers
        ArgumentCaptor<HttpRequest> requestCaptor = ArgumentCaptor.forClass(HttpRequest.class);
        verify(mockHttpClient).send(requestCaptor.capture(), any(HttpResponse.BodyHandler.class));
        HttpRequest executedRequest = requestCaptor.getValue();

        assertTrue(executedRequest.headers().firstValue("CB-ACCESS-KEY").isPresent());
        assertEquals("testApiKey", executedRequest.headers().firstValue("CB-ACCESS-KEY").get());
        assertTrue(executedRequest.headers().firstValue("CB-ACCESS-SIGN").isPresent());
        assertTrue(executedRequest.headers().firstValue("CB-ACCESS-TIMESTAMP").isPresent());
        assertTrue(executedRequest.headers().firstValue("Content-Type").isPresent());
        assertEquals("application/json", executedRequest.headers().firstValue("Content-Type").get());
        assertEquals(URI.create("https://api.coinbase.com/api/v3/brokerage/accounts"), executedRequest.uri());
    }

    @Test
    @SuppressWarnings("unchecked")
    void getAccountBalances_emptyResponse() throws CoinbaseApiException, IOException, InterruptedException {
        String mockJsonResponse = "{\"accounts\": []}";
        when(mockHttpResponse.statusCode()).thenReturn(200);
        when(mockHttpResponse.body()).thenReturn(mockJsonResponse);
        when(mockHttpClient.send(any(HttpRequest.class), any(HttpResponse.BodyHandler.class)))
                .thenReturn(mockHttpResponse);

        List<Map<String, Object>> balances = apiClient.getAccountBalances();
        assertNotNull(balances);
        assertTrue(balances.isEmpty());
    }

    @Test
    @SuppressWarnings("unchecked")
    void getAccountBalances_unexpectedJsonStructure() throws IOException, InterruptedException {
        String mockJsonResponse = "{\"unexpected_key\": []}";
        when(mockHttpResponse.statusCode()).thenReturn(200);
        when(mockHttpResponse.body()).thenReturn(mockJsonResponse);
        when(mockHttpClient.send(any(HttpRequest.class), any(HttpResponse.BodyHandler.class)))
                .thenReturn(mockHttpResponse);

        CoinbaseApiException exception = assertThrows(CoinbaseApiException.class, () -> apiClient.getAccountBalances());
        assertTrue(exception.getMessage().contains("Unexpected JSON structure from Coinbase API"));
    }

    @Test
    @SuppressWarnings("unchecked")
    void getAccountBalances_httpError4xx() throws IOException, InterruptedException {
        String errorJson = "{\"error\":\"INVALID_API_KEY\",\"error_details\":\"Invalid API Key\",\"message\":\"Invalid API Key\"}";
        when(mockHttpResponse.statusCode()).thenReturn(401); // Unauthorized
        when(mockHttpResponse.body()).thenReturn(errorJson);
        when(mockHttpClient.send(any(HttpRequest.class), any(HttpResponse.BodyHandler.class)))
                .thenReturn(mockHttpResponse);

        CoinbaseApiException exception = assertThrows(CoinbaseApiException.class, () -> apiClient.getAccountBalances());
        assertTrue(exception.getMessage().contains("Coinbase API request failed: 401"));
        assertTrue(exception.getMessage().contains("INVALID_API_KEY"));
    }

    @Test
    @SuppressWarnings("unchecked")
    void getAccountBalances_httpError5xx() throws IOException, InterruptedException {
        String errorJson = "{\"error\":\"INTERNAL_SERVER_ERROR\",\"message\":\"Internal Server Error\"}";
        when(mockHttpResponse.statusCode()).thenReturn(500); // Internal Server Error
        when(mockHttpResponse.body()).thenReturn(errorJson);
        when(mockHttpClient.send(any(HttpRequest.class), any(HttpResponse.BodyHandler.class)))
                .thenReturn(mockHttpResponse);

        CoinbaseApiException exception = assertThrows(CoinbaseApiException.class, () -> apiClient.getAccountBalances());
        assertTrue(exception.getMessage().contains("Coinbase API request failed: 500"));
        assertTrue(exception.getMessage().contains("INTERNAL_SERVER_ERROR"));
    }

    @Test
    void getAccountBalances_httpClientThrowsIOException() throws IOException, InterruptedException {
        when(mockHttpClient.send(any(HttpRequest.class), any(HttpResponse.BodyHandler.class)))
                .thenThrow(new IOException("Network issue"));

        CoinbaseApiException exception = assertThrows(CoinbaseApiException.class, () -> apiClient.getAccountBalances());
        assertTrue(exception.getMessage().contains("Network or IO error calling Coinbase API: Network issue"));
    }

    @Test
    void getAccountBalances_httpClientThrowsInterruptedException() throws IOException, InterruptedException {
        when(mockHttpClient.send(any(HttpRequest.class), any(HttpResponse.BodyHandler.class)))
                .thenThrow(new InterruptedException("Call interrupted"));

        CoinbaseApiException exception = assertThrows(CoinbaseApiException.class, () -> apiClient.getAccountBalances());
        assertTrue(exception.getMessage().contains("Coinbase API call interrupted: Call interrupted"));
        assertTrue(Thread.currentThread().isInterrupted()); // Check if interrupt flag is restored
    }

    @Test
    void generateSignature_producesNonEmptyString() throws CoinbaseApiException {
        // This is a basic test for the signature generation, not a cryptographic verification.
        // A more thorough test would require known inputs and outputs or a way to verify against Coinbase's examples.
        CoinbaseApiClient localClient = new CoinbaseApiClient(mockApiConfig, mockHttpClient); // use local instance for direct method call
        String timestamp = String.valueOf(Instant.now().getEpochSecond());
        String signature = localClient.generateSignature(timestamp, "GET", "/api/v3/brokerage/accounts", "");
        assertNotNull(signature);
        assertFalse(signature.isEmpty());
        // Signature should be hex encoded SHA256, typically 64 characters
        assertEquals(64, signature.length());
    }
}
