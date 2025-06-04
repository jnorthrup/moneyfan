@file:Repository("https://maven.pkg.jetbrains.space/public/p/ktor/eap")
@file:Repository("https://repo1.maven.org/maven2/")
@file:DependsOn("io.ktor:ktor-client-core:2.3.9")
@file:DependsOn("io.ktor:ktor-client-cio:2.3.9")
@file:DependsOn("io.ktor:ktor-client-content-negotiation:2.3.9")
@file:DependsOn("io.ktor:ktor-serialization-kotlinx-json:2.3.9")
@file:DependsOn("io.ktor:ktor-client-logging:2.3.9")
@file:DependsOn("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.8.0")
@file:DependsOn("org.jetbrains.kotlinx:kotlinx-serialization-json:1.6.3")
@file:DependsOn("commons-codec:commons-codec:1.16.0")

import kotlinx.coroutines.*
import kotlinx.serialization.*
import kotlinx.serialization.json.*
import io.ktor.client.*
import io.ktor.client.engine.cio.*
import io.ktor.client.plugins.contentnegotiation.*
import io.ktor.client.plugins.logging.*
import io.ktor.client.request.*
import io.ktor.client.statement.*
import io.ktor.http.*
import io.ktor.serialization.kotlinx.json.*
import java.nio.charset.StandardCharsets
import javax.crypto.Mac
import javax.crypto.spec.SecretKeySpec
import java.time.Instant
import java.util.UUID
import org.apache.commons.codec.binary.Hex

// --- Data Classes ---

// Changed to a regular class
class CoinbaseApiException(override val message: String, val statusCode: Int? = null, val responseBody: String? = null, override val cause: Throwable? = null) : RuntimeException(message, cause)

@Serializable
data class BalanceValue(val value: String, val currency: String)

@Serializable
data class Account( // Renamed from AccountBalance for clarity, as this is the raw account object
    val uuid: String,
    val name: String,
    val currency: String,
    @SerialName("available_balance") val availableBalance: BalanceValue,
    val default: Boolean,
    val active: Boolean,
    val hold: BalanceValue, // This is the hold balance object from Coinbase
    @SerialName("account_type") val accountType: String? = null, // Added from common structure, optional
    @SerialName("created_at") val createdAt: String? = null, // Optional ISO8601 timestamp
    @SerialName("updated_at") val updatedAt: String? = null, // Optional ISO8601 timestamp
    @SerialName("deleted_at") val deletedAt: String? = null, // Optional ISO8601 timestamp
    @SerialName("ready") val ready: Boolean? = null, // For retail advanced this is present
    @SerialName("type") val typeLegacy: String? = null // From older APIs, might appear, good to ignore
)

@Serializable
data class AccountsResponse( // Wrapper object if API returns {"accounts": [...]}
    val accounts: List<Account>,
    @SerialName("has_next") val hasNext: Boolean? = null, // Common pagination field
    val cursor: String? = null,
    val size: Int? = null
)

@Serializable
data class ProductDetails(
    @SerialName("product_id") val productId: String,
    val price: String, // The current market price for the product
    @SerialName("price_percentage_change_24h") val pricePercentageChange24h: String,
    val volume_24h: String, // Total volume traded in the last 24 hours for this product
    @SerialName("volume_percentage_change_24h") val volumePercentageChange24h: String,
    @SerialName("base_increment") val baseIncrement: String,
    @SerialName("quote_increment") val quoteIncrement: String,
    @SerialName("quote_min_size") val quoteMinSize: String, // Minimum size for quote currency
    @SerialName("quote_max_size") val quoteMaxSize: String, // Maximum size for quote currency
    @SerialName("base_min_size") val baseMinSize: String,  // Minimum size for base currency
    @SerialName("base_max_size") val baseMaxSize: String,  // Maximum size for base currency
    @SerialName("base_name") val baseName: String,
    @SerialName("quote_name") val quoteName: String,
    val watched: Boolean,
    @SerialName("is_disabled") val isDisabled: Boolean,
    @SerialName("new") val isNew: Boolean, // "new" is a keyword in Kotlin, kotlinx.serialization handles it
    val status: String, // e.g. "online", "offline", "internal_delisted", "delisted"
    val cancel_only: Boolean,
    val limit_only: Boolean,
    val post_only: Boolean,
    val trading_disabled: Boolean,
    val auction_mode: Boolean,
    @SerialName("product_type") val productType: String, // e.g. "SPOT"
    @SerialName("quote_currency_id") val quoteCurrencyId: String,
    @SerialName("base_currency_id") val baseCurrencyId: String,
    // Fields for futures
    @SerialName("fcm_trading_session_details") val fcmTradingSessionDetails: JsonObject? = null, // If any, or specific DTO
    @SerialName("mid_market_price") val midMarketPrice: String? = null,
    @SerialName("alias_to") val aliasTo: List<String>? = null, // For futures, e.g. ["ETH-PERP"]
    @SerialName("base_display_symbol") val baseDisplaySymbol: String? = null,
    @SerialName("quote_display_symbol") val quoteDisplaySymbol: String? = null,
    @SerialName("view_only") val viewOnly: Boolean? = null,
    @SerialName("min_market_funds") val minMarketFunds: String? = null, // Used in task description
    @SerialName("max_market_funds") val maxMarketFunds: String? = null  // Used in task description
)

@Serializable
data class OrderConfigurationMarket( // For Market orders
    @SerialName("quote_size") val quoteSize: String? = null, // For market BUY (amount of quote currency)
    @SerialName("base_size") val baseSize: String? = null    // For market SELL (amount of base currency)
)

@Serializable
data class CreateOrderRequest(
    @SerialName("client_order_id") val clientOrderId: String,
    @SerialName("product_id") val productId: String,
    val side: String, // "BUY" or "SELL"
    @SerialName("order_configuration") val orderConfiguration: OrderConfigurationMarket
)

@Serializable
data class OrderResponse( // Simplified, actual response is more complex
    val success: Boolean,
    @SerialName("failure_reason") val failureReason: String? = null,
    @SerialName("order_id") val orderId: String,
    @SerialName("success_response") val successResponse: OrderSuccessResponse? = null,
    @SerialName("error_response") val errorResponse: OrderErrorResponse? = null
) {
    @Serializable
    data class OrderSuccessResponse(
        @SerialName("order_id") val orderId: String,
        @SerialName("product_id") val productId: String,
        val side: String,
        @SerialName("client_order_id") val clientOrderId: String
    )
    @Serializable
    data class OrderErrorResponse(
        val error: String? = null,
        val message: String? = null,
        @SerialName("error_details") val errorDetails: String? = null,
        @SerialName("preview_failure_reason") val previewFailureReason: String? = null,
        @SerialName("new_order_failure_reason") val newOrderFailureReason: String? = null
    )
}


// --- Coinbase API Client ---

class CoinbaseApi(
    private val apiKey: String,
    apiSecret: String
) {
    private val apiSecretBytes = apiSecret.toByteArray(StandardCharsets.UTF_8)
    private val json = Json {
        ignoreUnknownKeys = true
        isLenient = true
        // prettyPrint = true // Ktor logging plugin will handle pretty printing if body logging is enabled
    }

    private val httpClient = HttpClient(CIO) {
        expectSuccess = false // Handle HTTP errors manually
        install(ContentNegotiation) {
            json(json) // Register Kotlinx.serialization converter
        }
        install(Logging) {
            logger = Logger.DEFAULT
            level = LogLevel.HEADERS // LogLevel.BODY for full body, HEADERS for less verbosity
            // filter { request -> request.url.host.contains("coinbase.com") } // Example filter
        }
    }

    private suspend fun signRequest(method: String, requestPath: String, body: String = ""): Triple<String, String, String> {
        val timestamp = Instant.now().epochSecond.toString()
        val prehash = timestamp + method.uppercase() + requestPath + body

        try {
            val mac = Mac.getInstance("HmacSHA256")
            val secretKeySpec = SecretKeySpec(apiSecretBytes, "HmacSHA256")
            mac.init(secretKeySpec)
            val signatureBytes = mac.doFinal(prehash.toByteArray(StandardCharsets.UTF_8))
            val signatureHex = Hex.encodeHexString(signatureBytes)
            return Triple(timestamp, signatureHex, prehash)
        } catch (e: Exception) {
            throw CoinbaseApiException("Error generating signature: ${e.message}", cause = e)
        }
    }

    private suspend inline fun <reified TResponse, reified TRequest> makeRequest(
        method: HttpMethod,
        path: String,
        requestBodyObject: TRequest? = null
    ): TResponse {
        val bodyString = requestBodyObject?.let { json.encodeToString(serializer<TRequest>(), it) } ?: ""
        val (timestamp, signature, _) = signRequest(method.value, path, bodyString)

        val response: HttpResponse = httpClient.request(CoinbaseApi.API_BASE_URL + path) {
            this.method = method
            headers {
                append("CB-ACCESS-KEY", apiKey)
                append("CB-ACCESS-SIGN", signature)
                append("CB-ACCESS-TIMESTAMP", timestamp)
                append(HttpHeaders.ContentType, ContentType.Application.Json)
            }
            if (requestBodyObject != null && bodyString.isNotEmpty()) { // Ktor setBody(null) might error
                setBody(requestBodyObject) // Ktor will use ContentNegotiation to serialize
            } else if (method == HttpMethod.Post || method == HttpMethod.Put) { // Ensure empty body for POST/PUT if no object
                setBody("")
            }
        }

        val responseBodyText = response.bodyAsText()
        if (response.status.isSuccess()) {
            try {
                return json.decodeFromString<TResponse>(responseBodyText)
            } catch (e: SerializationException) {
                throw CoinbaseApiException("Failed to deserialize successful response: ${e.message}. Body: $responseBodyText", statusCode = response.status.value, responseBody = responseBodyText, cause = e)
            }
        } else {
            throw CoinbaseApiException(
                message = "Coinbase API Error: ${response.status.value} ${response.status.description}. Body: $responseBodyText",
                statusCode = response.status.value,
                responseBody = responseBodyText
            )
        }
    }

    // Public API functions
    suspend fun getAccountBalances(): List<Account> {
        val responseWrapper = makeRequest<AccountsResponse, Unit>(HttpMethod.Get, ACCOUNTS_ENDPOINT_PATH)
        return responseWrapper.accounts
    }

    suspend fun getProductDetails(productId: String): ProductDetails {
        return makeRequest<ProductDetails, Unit>(HttpMethod.Get, "$PRODUCTS_ENDPOINT_PATH/$productId")
    }

    suspend fun placeMarketOrder(productId: String, side: String, sizeOrFunds: String): OrderResponse {
        val clientOrderId = UUID.randomUUID().toString()
        val orderConfiguration = when (side.uppercase()) {
            "BUY" -> OrderConfigurationMarket(quoteSize = sizeOrFunds)
            "SELL" -> OrderConfigurationMarket(baseSize = sizeOrFunds)
            else -> throw IllegalArgumentException("Side must be BUY or SELL. Was: $side")
        }
        val orderRequest = CreateOrderRequest(
            clientOrderId = clientOrderId,
            productId = productId,
            side = side.uppercase(),
            orderConfiguration = orderConfiguration
        )
        return makeRequest<OrderResponse, CreateOrderRequest>(HttpMethod.Post, ORDERS_ENDPOINT_PATH, orderRequest)
    }

    companion object {
        private const val API_BASE_URL = "https://api.coinbase.com"
        private const val ACCOUNTS_ENDPOINT_PATH = "/api/v3/brokerage/accounts"
        private const val PRODUCTS_ENDPOINT_PATH = "/api/v3/brokerage/products"
        private const val ORDERS_ENDPOINT_PATH = "/api/v3/brokerage/orders"

        fun createFromEnv(): CoinbaseApi {
            val apiKey = System.getenv("COINBASE_API_KEY")
                ?: throw IllegalStateException("COINBASE_API_KEY environment variable not set.")
            val apiSecret = System.getenv("COINBASE_API_SECRET")
                ?: throw IllegalStateException("COINBASE_API_SECRET environment variable not set.")
            return CoinbaseApi(apiKey, apiSecret)
        }
    }
}

// --- Main function for demonstration (optional) ---
fun main() = runBlocking {
    println("Coinbase Ktor Bot Script - Ktor: ${io.ktor.client.Utils.KTOR_VERSION}")
    val ktorLogger = io.ktor.client.plugins.logging.Logger.DEFAULT // Ktor's default console logger

    try {
        val apiClient = CoinbaseApi.createFromEnv()

        ktorLogger.info("Fetching account balances...")
        val balances = apiClient.getAccountBalances()
        ktorLogger.info("Fetched ${balances.size} accounts.")
        balances.take(5).forEach { ktorLogger.info(Json.encodeToString(it)) } // Print first few

        // Example: Get product details (replace with a valid product ID like "BTC-USD")
        val sampleProductId = "BTC-USD"
        ktorLogger.info("\\nFetching $sampleProductId product details...")
        try {
            val product = apiClient.getProductDetails(sampleProductId)
            ktorLogger.info(Json.encodeToString(product))
        } catch (e: CoinbaseApiException) {
            ktorLogger.error("Error fetching product $sampleProductId: ${e.message} - Status: ${e.statusCode}, Body: ${e.responseBody}")
        }

        // Example: Place a test market order (USE WITH EXTREME CAUTION - REAL MONEY)
        // This part should remain commented out or used with a sandbox API key if available.
        /*
        try {
            ktorLogger.info("\\nPlacing a test market BUY order for BTC-USD (with 1 USD)...")
            // Ensure the product_id and size/funds are appropriate for your test account and comply with min_market_funds.
            // Check product.minMarketFunds before running.
            val buyOrderResponse = apiClient.placeMarketOrder("BTC-USD", "BUY", "1.00") // Example: 1 USD worth of BTC
            ktorLogger.info("Buy Order Response: ${Json.encodeToString(buyOrderResponse)}")

        } catch (e: CoinbaseApiException) {
            ktorLogger.error("Error placing order: ${e.message} - Status: ${e.statusCode}, Body: ${e.responseBody}")
        }
        */

    } catch (e: IllegalStateException) {
        ktorLogger.error("Configuration error: ${e.message}")
    } catch (e: CoinbaseApiException) {
        ktorLogger.error("Coinbase API Error: ${e.message} - Status: ${e.statusCode}, Body: ${e.responseBody}")
        e.cause?.let { ktorLogger.error("Cause: ${it.message}") }
    } catch (e: Exception) { // Catch-all for other unexpected errors
        ktorLogger.error("An unexpected error occurred in main: ${e.message}")
        e.printStackTrace()
    }
}
```
