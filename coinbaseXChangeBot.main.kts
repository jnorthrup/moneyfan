@file:Repository("https://repo1.maven.org/maven2/")
// Core XChange
@file:DependsOn("org.knowm.xchange:xchange-core:5.1.1")

// Coinbase Exchange Module
@file:DependsOn("org.knowm.xchange:xchange-coinbasepro:5.1.1")

// SLF4J for logging (XChange uses it)
@file:DependsOn("org.slf4j:slf4j-api:2.0.9")
@file:DependsOn("org.slf4j:slf4j-simple:2.0.9")

// Kotlinx Coroutines
@file:DependsOn("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.7.3")

// Kotlinx Serialization
@file:DependsOn("org.jetbrains.kotlinx:kotlinx-serialization-json:1.6.3")

// Required Jackson dependencies for XChange
@file:DependsOn("com.fasterxml.jackson.core:jackson-core:2.15.2")
@file:DependsOn("com.fasterxml.jackson.core:jackson-annotations:2.15.2")
@file:DependsOn("com.fasterxml.jackson.core:jackson-databind:2.15.2")
@file:DependsOn("com.fasterxml.jackson.datatype:jackson-datatype-jsr310:2.15.2")

// RxJava for Streaming
@file:DependsOn("io.reactivex.rxjava3:rxjava:3.1.8")

/**
 * Coinbase Trading Bot (Ported from JS Kraken Bot) - Kotlin Script (.kts)
 *
 * Description:
 * This script implements an automated trading bot for Coinbase, based on the logic
 * of a JavaScript Kraken bot. It uses the Knowm XChange library for interacting
 * with the Coinbase Advanced Trade API.
 *
 * Features:
 * - Connects to Coinbase API (API Key & Secret required).
 * - Fetches account balances and real-time market data via WebSockets.
 * - Persists trading state (baselines, trailing/rebalance states, timestamps) to JSON.
 * - Implements trading strategies:
 *   - Individual Asset Harvest (with forced harvest).
 *   - Portfolio Override Harvest (baseline reset).
 *   - Harvest Proceeds Allocation (reinvest, BTC buy, cash).
 *   - Rebalancing (standard and forced).
 *   - Adaptive Dead Zone (ADZ) for harvest/rebalance triggers.
 *   - Portfolio Crash Protection (CP) to adjust strategy parameters.
 *
 * Setup:
 * 1. Ensure Kotlin is installed and accessible in your PATH (or use a Kotlin JSR223 runner).
 * 2. Set the following environment variables:
 *    - COINBASE_API_KEY: Your Coinbase API Key.
 *    - COINBASE_API_SECRET: Your Coinbase API Secret.
 *    - COINBASE_PASSPHRASE: Your Coinbase API Passphrase (if using xchange-coinbasepro and your key requires it).
 *    (Ensure the API key has permissions for viewing balances, market data, and trading).
 *
 * Running the Script:
 *   kotlin -script coinbaseXChangeBot.main.kts
 *
 * Disclaimer:
 * TRADING CRYPTOCURRENCIES IS RISKY. THIS SCRIPT IS FOR EDUCATIONAL AND
 * EXPERIMENTAL PURPOSES ONLY. USE AT YOUR OWN RISK. THE CREATORS AND
 * CONTRIBUTORS ARE NOT RESPONSIBLE FOR ANY FINANCIAL LOSSES.
 * Simulated order placement is used by default. Modify with caution.
 */

import kotlinx.coroutines.*
import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.*
import org.knowm.xchange.*
import org.knowm.xchange.coinbasepro.CoinbaseProExchange
import org.knowm.xchange.currency.Currency
import org.knowm.xchange.currency.CurrencyPair
import org.knowm.xchange.dto.Order
import org.knowm.xchange.dto.account.Balance
import org.knowm.xchange.dto.marketdata.Ticker
import org.knowm.xchange.dto.meta.CurrencyMetaData
import org.knowm.xchange.dto.meta.CurrencyPairMetaData
import org.knowm.xchange.dto.meta.ExchangeMetaData
import org.knowm.xchange.dto.trade.MarketOrder
import org.knowm.xchange.service.account.AccountService
import org.knowm.xchange.service.marketdata.MarketDataService
import org.knowm.xchange.service.trade.TradeService
import org.knowm.xchange.streaming.StreamingExchange
import org.knowm.xchange.streaming.StreamingMarketDataService
import org.slf4j.LoggerFactory
import java.math.BigDecimal
import java.math.MathContext
import java.math.RoundingMode
import io.reactivex.rxjava3.disposables.CompositeDisposable
import io.reactivex.rxjava3.disposables.Disposable
import java.io.File
import java.time.Instant
import java.time.Duration // For cycle timing
import kotlin.math.max


// --- Bot State Data Classes ---
@Serializable
data class TrailingData(
    val flagged: Boolean = false,
    val harvestCycleCount: Int = 0,
    val flaggedAt: Long? = null,
    val previousDeviation: Double? = null
)

@Serializable
data class RebalanceData(
    val triggered: Boolean = false,
    val triggeredAt: Long? = null,
    var rebalancePosCycleCount: Int = 0,
    var attemptCount: Int = 0,
    var cooldownUntil: Long = 0L,
    val currentBaselineWhenTriggered: Double? = null,
    var previousDeviation: Double? = null
)

@Serializable
data class BotState(
    val baselines: MutableMap<String, Double> = mutableMapOf(),
    val trailingState: MutableMap<String, TrailingData> = mutableMapOf(),
    val lastActionTimestamps: MutableMap<String, Long> = mutableMapOf(),
    val rebalanceState: MutableMap<String, RebalanceData> = mutableMapOf(),
    val adaptiveDeadZoneState: MutableMap<String, Boolean> = mutableMapOf() // Symbol -> IsADZActive
)

// --- Global State Variable ---
lateinit var botState: BotState

// --- Strategy Constants and Top-Level Variables ---
val QUOTE_CURRENCY_CODE = "USD"
val QUOTE_CURRENCY = Currency(QUOTE_CURRENCY_CODE)

// --- Individual Asset Harvest ---
val HARVEST_EXCLUDE_SYMBOLS = setOf("BTC", "USDC", QUOTE_CURRENCY_CODE)
const val FLAT_HARVEST_TRIGGER_PERCENT = 0.03
const val HARVEST_CYCLE_THRESHOLD = 3
const val MIN_SURPLUS_FOR_HARVEST = 1.00 // USD
const val MIN_SURPLUS_FOR_FORCED_HARVEST = 1.00 // USD
const val FORCED_HARVEST_TIMEOUT = 20 * 60 * 1000L
const val TARGET_ADJUST_PERCENT = 0.000

// --- Portfolio Override Harvest (Baseline Reset) ---
const val ENABLE_PORTFOLIO_HARVEST = true
const val PORTFOLIO_HARVEST_TRIGGER_DEVIATION_PERCENT = 0.05
const val PORTFOLIO_HARVEST_CONFIRMATION_CYCLES = 3
const val MIN_ASSET_SURPLUS_FOR_PORTFOLIO_HARVEST = 0.10 // USD
val REBALANCE_EXCLUDE_SYMBOLS = setOf("USDC", QUOTE_CURRENCY_CODE) // For portfolio deviation calculation

// --- Harvest Proceeds Allocation ---
const val HARVEST_ALLOC_REINVEST_PERCENT = 0.50
const val HARVEST_ALLOC_CASH_PERCENT = 0.40
const val HARVEST_ALLOC_BTC_PERCENT = 0.10
const val MIN_HARVEST_TO_ALLOCATE = 1.00 // USD
const val MIN_NEGATIVE_DEVIATION_FOR_REINVEST = -0.01 // -1%
const val MIN_REINVEST_BUY_USD = 0.50
const val MIN_BTC_BUY_USD = 1.00

// --- Rebalance ---
// REBALANCE_EXCLUDE_SYMBOLS already defined above, used for both portfolio deviation and rebalance exclusion
const val FLAT_REBALANCE_TRIGGER_PERCENT = 0.04
const val PARTIAL_RECOVERY_PERCENT = 0.875
const val REBALANCE_POSITIVE_THRESHOLD = 3
const val MAX_REBALANCE_ATTEMPTS = 3
const val REBALANCE_COOLDOWN = 30 * 60 * 1000L
const val FORCE_REBALANCE_TIMEOUT = 25 * 60 * 1000L
const val FORCE_REBALANCE_SHORTFALL_PERCENT = 0.25
const val MIN_PARTIAL_REBALANCE_USD = 1.00
const val MIN_FORCED_REBALANCE_USD = 1.00

// --- Adaptive Dead Zone Mode ---
const val ENABLE_ADAPTIVE_DEAD_ZONE = true
const val ADAPTIVE_DZ_INACTIVITY_TIMEOUT = 3 * 60 * 60 * 1000L // 3 hours
const val ADAPTIVE_DZ_HARVEST_TRIGGER_PERCENT = 0.020 // +2.0%
const val ADAPTIVE_DZ_REBALANCE_TRIGGER_PERCENT = 0.020 // -2.0%

// --- Portfolio-Level Crash Protection ---
const val ENABLE_CRASH_PROTECTION = true
const val CP_TRIGGER_ASSET_PERCENT = 0.70 // 70% of assets
const val CP_TRIGGER_MIN_NEGATIVE_DEV_PERCENT = -0.01 // -1%
const val CRASH_PROTECTION_THRESHOLD_INCREASE = 2 // For rebalance cycles
const val CRASH_PROTECTION_PARTIAL_RECOVERY_PERCENT_FACTOR = 0.55 / 0.875


val latestPrices = mutableMapOf<CurrencyPair, Ticker>()
val assetExchangeMetaData = mutableMapOf<Currency, CurrencyMetaData?>()
val assetPairMetaData = mutableMapOf<CurrencyPair, CurrencyPairMetaData?>()

var initialized = false
val mainLoopLogger = LoggerFactory.getLogger("MainLoopLogic")

data class PortfolioRow(
    val symbol: String,
    val currency: Currency,
    val quantity: BigDecimal,
    val price: BigDecimal?,
    val value: BigDecimal?,
    val baseline: Double?,
    val deviation: Double?,
    val absoluteDifference: BigDecimal?
)

data class PortfolioHarvestStateData(
    var flagged: Boolean = false,
    var cycleCount: Int = 0,
    var flaggedAt: Long? = null,
    var previousDeviationPercent: Double? = null
)
var portfolioHarvestState = PortfolioHarvestStateData()

var harvestedAmountThisCycle = BigDecimal.ZERO
var anyTradesThisCycle = false


// --- Graceful Shutdown Hook ---
val shutdownLogger = LoggerFactory.getLogger("ShutdownHook")
val shutdownHook = Thread {
    shutdownLogger.info("Process termination detected. Saving state...")
    if (::botState.isInitialized) {
        try {
            // Simplified save, no complex file operations, assuming botState is consistent enough
            val json = Json { prettyPrint = true; encodeDefaults = true; ignoreUnknownKeys = true }
            val jsonString = json.encodeToString(botState)
            File(StateManager.STATE_FILE_PATH).writeText(jsonString) // Overwrite directly
            shutdownLogger.info("Bot state saved to ${StateManager.STATE_FILE_PATH} during shutdown.")
        } catch (e: Exception) {
            shutdownLogger.error("CRITICAL ERROR: Failed to save state during shutdown: ${e.message}", e)
        }
    } else {
        shutdownLogger.info("botState not initialized, no state to save during shutdown.")
    }

    if (ExchangeService.isExchangeInitialized()) { // Need a way to check this
         ExchangeService.cleanup() // Call existing cleanup
         shutdownLogger.info("ExchangeService cleanup called during shutdown.")
    }
    shutdownLogger.info("Shutdown hook finished.")
}
// Register the shutdown hook
// Runtime.getRuntime().addShutdownHook(shutdownHook) // Will be registered in main


// --- Helper Functions ---
fun roundQuantity(
    currency: Currency,
    pair: CurrencyPair,
    quantity: BigDecimal
): BigDecimal {
    val pairMeta = assetPairMetaData[pair]
    val currencyMeta = assetExchangeMetaData[currency]

    val minAmount = pairMeta?.minimumAmount ?: BigDecimal.ZERO

    val scale = pairMeta?.baseScale ?: currencyMeta?.scale ?: 8

    if (quantity.compareTo(BigDecimal.ZERO) == 0) return BigDecimal.ZERO

    val roundedQty = quantity.setScale(scale, RoundingMode.FLOOR)

    if (roundedQty.compareTo(BigDecimal.ZERO) > 0 && roundedQty < minAmount) {
        mainLoopLogger.warn("Rounded quantity $roundedQty for ${currency.currencyCode} is below minimum $minAmount for pair $pair. Returning ZERO.")
        return BigDecimal.ZERO
    }
    if (roundedQty.compareTo(BigDecimal.ZERO) < 0) {
         return BigDecimal.ZERO
    }
    return roundedQty
}

fun logTrade(asset: String, side: String, quantity: String, price: String, orderId: String?, note: String) {
    mainLoopLogger.info("TRADE: $side $quantity $asset @ ~$price (Order ID: ${orderId ?: "N/A"}) - Note: $note")
}


// --- State Manager Object ---
object StateManager {
    private val logger = LoggerFactory.getLogger(StateManager::class.java)
    private val json = Json {
        prettyPrint = true
        encodeDefaults = true
        isLenient = true
        ignoreUnknownKeys = true
    }
    const val STATE_FILE_PATH = "coinbaseBotState.json"

    fun loadState(): BotState {
        val stateFile = File(STATE_FILE_PATH)
        if (stateFile.exists() && stateFile.canRead()) {
            try {
                val data = stateFile.readText()
                if (data.isNotBlank()) {
                    val loaded = json.decodeFromString<BotState>(data)
                    logger.info("Successfully loaded state from $STATE_FILE_PATH")
                    return BotState(
                        baselines = loaded.baselines.toMutableMap(),
                        trailingState = loaded.trailingState.toMutableMap(),
                        lastActionTimestamps = loaded.lastActionTimestamps.toMutableMap(),
                        rebalanceState = loaded.rebalanceState.toMutableMap(),
                        adaptiveDeadZoneState = loaded.adaptiveDeadZoneState.toMutableMap()
                    )
                } else {
                     logger.warn("State file $STATE_FILE_PATH is empty. Starting with default state.")
                    return BotState()
                }
            } catch (e: Exception) {
                logger.error("Error loading state from $STATE_FILE_PATH. File might be corrupted or incompatible. Starting with default state. Error: ${e.message}", e)
                return BotState()
            }
        } else {
            logger.info("State file $STATE_FILE_PATH not found or not readable. Starting with default state.")
            return BotState()
        }
    }

    fun saveState(state: BotState) {
        val tempFilePath = "$STATE_FILE_PATH.tmp"
        val tempFile = File(tempFilePath)
        val finalFile = File(STATE_FILE_PATH)
        try {
            val jsonString = json.encodeToString(state)
            tempFile.writeText(jsonString)

            if (finalFile.exists()) {
                if (!finalFile.delete()) {
                    logger.warn("Could not delete old state file $STATE_FILE_PATH before rename.")
                }
            }

            if (tempFile.renameTo(finalFile)) {
                logger.info("Successfully saved state to $STATE_FILE_PATH")
            } else {
                logger.error("Failed to rename temp state file $tempFilePath to $STATE_FILE_PATH. Attempting copy as fallback.")
                try {
                    tempFile.copyTo(finalFile, overwrite = true)
                    logger.info("Successfully copied temp state file to $STATE_FILE_PATH as fallback.")
                    if (!tempFile.delete()) {
                        logger.warn("Failed to delete temp file $tempFilePath after fallback copy.")
                    }
                } catch (copyEx: Exception) {
                    logger.error("CRITICAL: Failed to copy temp state file to $STATE_FILE_PATH as fallback: ${copyEx.message}", copyEx)
                    logger.error("State was written to $tempFilePath but could not be moved to $STATE_FILE_PATH.")
                }
            }
        } catch (e: Exception) {
            logger.error("CRITICAL ERROR: Failed to save state (writing to temp file $tempFilePath). Error: ${e.message}", e)
        } finally {
             if (tempFile.exists() && finalFile.exists() && finalFile.length() > 0 && tempFile.readText() == finalFile.readText()) {
                tempFile.delete()
            } else if (tempFile.exists() && (!finalFile.exists() || finalFile.length() == 0L)) {
                logger.warn("Final state file $STATE_FILE_PATH might be missing or empty. Temp file $tempFilePath is being kept for safety.")
            }
        }
    }
}

sealed class OrderAmount {
    data class BaseSize(val amount: BigDecimal) : OrderAmount()
    data class QuoteSize(val amount: BigDecimal) : OrderAmount()
}

object ExchangeService {
    private val logger = LoggerFactory.getLogger(ExchangeService::class.java)
    private lateinit var exchangeVar: Exchange // Renamed to avoid conflict with lazy delegate
    private val disposables = CompositeDisposable()

    // Public way to check if exchange has been initialized
    fun isExchangeInitialized(): Boolean = this::exchangeVar.isInitialized


    fun initialize() {
        val apiKey = System.getenv("COINBASE_API_KEY")
        val secretKey = System.getenv("COINBASE_API_SECRET")

        if (apiKey.isNullOrBlank() || secretKey.isNullOrBlank()) {
            logger.error("API Key or Secret Key environment variables not set.")
            throw IllegalStateException("COINBASE_API_KEY and COINBASE_API_SECRET must be set.")
        }

        val exchangeSpecification = CoinbaseProExchange().defaultExchangeSpecification.apply {
            this.apiKey = apiKey
            this.secretKey = secretKey
            System.getenv("COINBASE_PASSPHRASE")?.let { if (it.isNotBlank()) this.passphrase = it }
            logger.info("Using Exchange: ${CoinbaseProExchange::class.java.name}")
        }
        exchangeVar = ExchangeFactory.INSTANCE.createExchange(exchangeSpecification) // Use renamed var
        logger.info("Exchange initialized: ${exchangeVar.exchangeSpecification.exchangeName}")

        try {
            logger.info("Attempting remoteInit to fetch exchange metadata...")
            val remoteMetaData = exchangeVar.remoteInit()
            if (remoteMetaData != null) {
                 logger.info("Exchange remoteInit successful. Currencies (sample): ${remoteMetaData.currencies?.keys?.take(10)}..., Pairs (sample): ${remoteMetaData.currencyPairs?.keys?.take(10)}...")
            } else {
                logger.warn("Remote metadata was null after remoteInit.")
            }
        } catch (e: Exception) {
            logger.error("Failed to initialize remote exchange metadata: ${e.message}", e)
        }
    }

    private val accountService: AccountService by lazy {
        if (!this::exchangeVar.isInitialized) throw IllegalStateException("ExchangeService not initialized.")
        exchangeVar.accountService
    }
    private val marketDataService: MarketDataService by lazy {
        if (!this::exchangeVar.isInitialized) throw IllegalStateException("ExchangeService not initialized.")
        exchangeVar.marketDataService
    }
    private val tradeService: TradeService by lazy {
        if (!this::exchangeVar.isInitialized) throw IllegalStateException("ExchangeService not initialized.")
        exchangeVar.tradeService
    }
    val exchangeMetaData: ExchangeMetaData? by lazy {
         if (!this::exchangeVar.isInitialized) throw IllegalStateException("ExchangeService not initialized.")
        exchangeVar.exchangeMetaData
    }

    suspend fun getAccountBalances(): Map<Currency, Balance>? {
        return withContext(Dispatchers.IO) {
            try {
                val accountInfo = accountService.accountInfo
                logger.debug("Fetched account info: $accountInfo")
                accountInfo?.getWallet()?.balances?.values?.associateBy { it.currency }
                    ?: accountInfo?.wallets?.values?.flatMap { it.balances.values }?.associateBy { it.currency }
            } catch (e: Exception) {
                logger.error("Error fetching account balances: ${e.message}", e)
                null
            }
        }
    }

    suspend fun getProductDetails(pair: CurrencyPair): CurrencyPairMetaData? {
        return withContext(Dispatchers.IO) {
            try {
                val metaData = exchangeMetaData ?: exchangeVar.remoteInit() // Use exchangeVar
                val details = metaData?.currencyPairs?.get(pair)
                logger.debug("Fetched product details for $pair: $details")
                details
            } catch (e: Exception) {
                logger.error("Error fetching product details for $pair: ${e.message}", e)
                null
            }
        }
    }

    suspend fun placeMarketOrder(
        pair: CurrencyPair,
        type: Order.OrderType,
        orderAmount: OrderAmount
    ): String? {
        return withContext(Dispatchers.IO) {
            try {
                val amountForOrder: BigDecimal
                when (type) {
                    Order.OrderType.BID -> {
                        when (orderAmount) {
                            is OrderAmount.QuoteSize -> {
                                logger.info("BUY order specified with QuoteSize: ${orderAmount.amount} ${pair.quoteCurrency}")
                                val ticker = marketDataService.getTicker(pair)
                                if (ticker?.last == null || ticker.last <= BigDecimal.ZERO) {
                                    logger.error("Could not fetch valid price ticker for $pair to calculate base size from quote size.")
                                    return@withContext null
                                }
                                val baseCurrencyScale = assetPairMetaData[pair]?.baseScale
                                    ?: assetExchangeMetaData[pair.base]?.scale
                                    ?: 8

                                amountForOrder = orderAmount.amount.divide(ticker.last, baseCurrencyScale, RoundingMode.DOWN)
                                logger.info("Calculated base size for BUY: $amountForOrder ${pair.baseCurrency} (from ${orderAmount.amount} ${pair.quoteCurrency} @ approx ${ticker.last})")
                                if (amountForOrder <= BigDecimal.ZERO) {
                                     logger.error("Calculated base amount $amountForOrder is zero or less. Market order not placed.")
                                     return@withContext null
                                }
                            }
                            is OrderAmount.BaseSize -> {
                                amountForOrder = orderAmount.amount
                                logger.info("BUY order specified with BaseSize: $amountForOrder ${pair.baseCurrency} (less common for market BUYs)")
                            }
                        }
                    }
                    Order.OrderType.ASK -> {
                        when (orderAmount) {
                            is OrderAmount.BaseSize -> {
                                amountForOrder = orderAmount.amount
                                logger.info("SELL order specified with BaseSize: $amountForOrder ${pair.baseCurrency}")
                            }
                            is OrderAmount.QuoteSize -> {
                                logger.warn("SELL order specified with QuoteSize is unusual for market orders. XChange expects base size for sells. Order not placed.")
                                return@withContext null
                            }
                        }
                    }
                    else -> {
                        logger.error("Unsupported order type: $type")
                        return@withContext null
                    }
                }

                val marketOrder = MarketOrder.Builder(type, pair)
                    .originalAmount(amountForOrder)
                    .build()
                logger.info("Placing market order via XChange: $marketOrder")
                val orderId = "sim_order_${System.currentTimeMillis()}" // tradeService.placeMarketOrder(marketOrder)
                logger.info("Market order placed successfully via XChange. Order ID: $orderId")
                orderId
            } catch (e: Exception) {
                logger.error("Error placing market order for $pair: ${e.message}", e)
                null
            }
        }
    }

    fun subscribeToPriceTicks(pair: CurrencyPair, onPriceUpdate: (Ticker) -> Unit): Disposable? {
        if (!this::exchangeVar.isInitialized || exchangeVar !is StreamingExchange) { // Use exchangeVar
            logger.warn("Streaming is not supported by ${exchangeVar.exchangeSpecification.exchangeName} or not initialized.")
            return null
        }
        val streamingExchange = exchangeVar as StreamingExchange
        if (!streamingExchange.isAlive) {
             logger.info("Streaming exchange is not alive. Attempting to connect...")
             try {
                streamingExchange.connect().blockingAwait()
                logger.info("Streaming exchange connected.")
             } catch (e: Exception) {
                 logger.error("Failed to connect streaming exchange: ${e.message}", e)
                 return null
             }
        }

        val streamingMarketDataService: StreamingMarketDataService = streamingExchange.streamingMarketDataService
        logger.info("Subscribing to ticker for $pair")

        val disposable = streamingMarketDataService.getTicker(pair)
            .subscribe(
                { ticker ->
                    onPriceUpdate(ticker)
                },
                { throwable -> logger.error("Error in ticker subscription for $pair: ${throwable.message}", throwable) },
                { logger.info("Ticker subscription for $pair completed.") }
            )
        disposables.add(disposable)
        return disposable
    }

    fun cleanup() {
        logger.info("Cleaning up ExchangeService resources...")
        disposables.clear()
        if (this::exchangeVar.isInitialized && exchangeVar is StreamingExchange) { // Use exchangeVar
            val streamingExchange = exchangeVar as StreamingExchange
            if (streamingExchange.isAlive) {
                logger.info("Disconnecting streaming exchange...")
                try {
                    streamingExchange.disconnect().blockingAwait()
                    logger.info("Streaming exchange disconnected.")
                } catch (e: Exception) {
                    logger.error("Error disconnecting streaming exchange: ${e.message}", e)
                }
            }
        }
        logger.info("ExchangeService cleanup complete.")
    }
}

fun main() = runBlocking {
    Runtime.getRuntime().addShutdownHook(shutdownHook) // Register shutdown hook

    mainLoopLogger.info("Coinbase XChange Bot Starting...")

    System.setProperty("org.slf4j.simpleLogger.defaultLogLevel", "info")
    System.setProperty("org.slf4j.simpleLogger.log.StateManager", "info")
    System.setProperty("org.slf4j.simpleLogger.log.ExchangeService", "info")
    System.setProperty("org.slf4j.simpleLogger.log.MainLoopLogic", "info")
    System.setProperty("org.slf4j.simpleLogger.log.org.knowm.xchange", "warn")
    System.setProperty("org.slf4j.simpleLogger.log.org.knowm.xchange.coinbasepro", "info")
    System.setProperty("org.slf4j.simpleLogger.showDateTime", "true")
    System.setProperty("org.slf4j.simpleLogger.dateTimeFormat", "yyyy-MM-dd HH:mm:ss:SSS Z")
    System.setProperty("org.slf4j.simpleLogger.showThreadName", "true")

    botState = StateManager.loadState()
    mainLoopLogger.info("Initial bot state loaded: ${botState.baselines.size} baselines, ${botState.trailingState.size} trailing states, ${botState.lastActionTimestamps.size} timestamps, ${botState.rebalanceState.size} rebalance states, ${botState.adaptiveDeadZoneState.size} ADZ states.")

    var activeSubscriptions = mutableMapOf<CurrencyPair, Disposable>()
    val REFRESH_INTERVAL = 8000L

    try {
        ExchangeService.initialize()
        mainLoopLogger.info("ExchangeService initialized.")

        while (true) {
            val cycleStartTime = System.currentTimeMillis() // Corrected to use System.currentTimeMillis()
            mainLoopLogger.info("----- Cycle Start: ${Instant.ofEpochMilli(cycleStartTime)} -----")
            harvestedAmountThisCycle = BigDecimal.ZERO
            anyTradesThisCycle = false
            var stateChangedThisCycle = false

            val accountBalances = ExchangeService.getAccountBalances()
            var cashBalance = BigDecimal.ZERO
            val currentHoldings = mutableMapOf<Currency, BigDecimal>()

            if (accountBalances == null) {
                mainLoopLogger.error("Failed to fetch account balances. Skipping cycle.")
                delay(REFRESH_INTERVAL)
                continue
            }

            accountBalances.forEach { (currency, balance) ->
                if (currency == QUOTE_CURRENCY) {
                    cashBalance = cashBalance.add(balance.available ?: BigDecimal.ZERO)
                } else if ((balance.available ?: BigDecimal.ZERO) > BigDecimal.ZERO || (balance.total ?: BigDecimal.ZERO) > BigDecimal.ZERO) {
                    currentHoldings[currency] = balance.total ?: BigDecimal.ZERO
                }
            }
            mainLoopLogger.info("Cash Balance: $QUOTE_CURRENCY_CODE ${cashBalance.toPlainString()}")
            if (currentHoldings.isNotEmpty()) {
                mainLoopLogger.info("Holdings: ${currentHoldings.entries.joinToString { it.key.currencyCode + ": " + it.value.toPlainString() }}")
            } else {
                mainLoopLogger.info("No significant crypto holdings found.")
            }

            val symbolsToTrack = currentHoldings.keys.toMutableSet()
            if (HARVEST_ALLOC_BTC_PERCENT > 0 && MIN_BTC_BUY_USD < 1000) {
                 symbolsToTrack.add(Currency.BTC)
            }

            val pairsToSubscribe = symbolsToTrack
                .filter { it != QUOTE_CURRENCY }
                .map { CurrencyPair(it, QUOTE_CURRENCY) }
                .toSet()

            val currentActivePairs = activeSubscriptions.keys.toSet()
            val pairsToUnsubscribe = currentActivePairs - pairsToSubscribe
            val newPairsToSubscribe = pairsToSubscribe - currentActivePairs

            pairsToUnsubscribe.forEach { pair ->
                activeSubscriptions.remove(pair)?.dispose()
                latestPrices.remove(pair)
                mainLoopLogger.info("Unsubscribed from ticker: $pair")
            }

            newPairsToSubscribe.forEach { pair ->
                val baseCurrency = pair.base
                if (!assetExchangeMetaData.containsKey(baseCurrency)) {
                    assetExchangeMetaData[baseCurrency] = ExchangeService.exchangeMetaData?.currencies?.get(baseCurrency)
                    mainLoopLogger.debug("Fetched metadata for currency $baseCurrency: ${assetExchangeMetaData[baseCurrency]}")
                }
                if (!assetPairMetaData.containsKey(pair)) {
                    assetPairMetaData[pair] = ExchangeService.getProductDetails(pair)
                    mainLoopLogger.debug("Fetched metadata for pair $pair: ${assetPairMetaData[pair]}")
                }

                val subscription = ExchangeService.subscribeToPriceTicks(pair) { ticker ->
                    latestPrices[ticker.currencyPair] = ticker
                }
                if (subscription != null) {
                    activeSubscriptions[pair] = subscription
                    mainLoopLogger.info("Subscribed to ticker: $pair")
                } else {
                    mainLoopLogger.warn("Failed to subscribe to ticker: $pair")
                }
            }
            if (newPairsToSubscribe.isNotEmpty() || pairsToUnsubscribe.isNotEmpty()) {
                 mainLoopLogger.info("Active ticker subscriptions: ${activeSubscriptions.keys.joinToString { it.toString() }}")
            }
            if (pairsToSubscribe.isNotEmpty() && newPairsToSubscribe.isNotEmpty()) {
                mainLoopLogger.info("Allowing 1s for new tickers to stream initial prices...")
                delay(1000)
            }

            var totalHoldingsValue = BigDecimal.ZERO
            val portfolioSummaryList = mutableListOf<PortfolioRow>()
            val currentSymbolsInPortfolio = mutableSetOf<String>()
            var baselinesVerifiedOrSetThisCycle = false
            val priceFetchIssues = mutableListOf<String>()

            currentHoldings.forEach { (currency, quantity) ->
                val pair = CurrencyPair(currency, QUOTE_CURRENCY)
                val priceTicker = latestPrices[pair]
                val currentPrice = priceTicker?.last

                if (currentPrice == null || currentPrice <= BigDecimal.ZERO) {
                    priceFetchIssues.add(currency.currencyCode)
                    return@forEach
                }

                currentSymbolsInPortfolio.add(currency.currencyCode)
                val currentHoldingValue = quantity.multiply(currentPrice)
                totalHoldingsValue = totalHoldingsValue.add(currentHoldingValue)
                val symbolCode = currency.currencyCode
                var baselineValue = botState.baselines[symbolCode]

                if (!initialized) {
                    if (baselineValue != null && baselineValue > 0.01) {
                        mainLoopLogger.info("✅ $symbolCode: Using loaded baseline $$baselineValue.")
                        baselinesVerifiedOrSetThisCycle = true
                    } else if (baselineValue == null && currentHoldingValue > BigDecimal.valueOf(0.01)) {
                        botState.baselines[symbolCode] = currentHoldingValue.toDouble()
                        baselineValue = currentHoldingValue.toDouble()
                        mainLoopLogger.info("✨ Initialized baseline $symbolCode: $$baselineValue (First cycle).")
                        baselinesVerifiedOrSetThisCycle = true
                        stateChangedThisCycle = true
                    }
                }

                if (initialized && baselineValue == null && currentHoldingValue > BigDecimal.valueOf(0.01)) {
                    botState.baselines[symbolCode] = currentHoldingValue.toDouble()
                    baselineValue = currentHoldingValue.toDouble()
                    mainLoopLogger.info("✨ Initialized baseline $symbolCode (post-init): $$baselineValue.")
                    stateChangedThisCycle = true
                }

                if (botState.lastActionTimestamps[symbolCode] == null && baselineValue != null && baselineValue > 0.01) {
                    botState.lastActionTimestamps[symbolCode] = System.currentTimeMillis()
                    mainLoopLogger.info("✨ Initialized last action timestamp for $symbolCode.")
                    stateChangedThisCycle = true
                }

                var deviation: Double? = null
                var absoluteDifference: BigDecimal? = null
                if (baselineValue != null && baselineValue > 0) {
                    val baselineBD = BigDecimal.valueOf(baselineValue)
                    absoluteDifference = currentHoldingValue.subtract(baselineBD)
                    if (baselineBD.compareTo(BigDecimal.ZERO) != 0) {
                       deviation = currentHoldingValue.subtract(baselineBD).divide(baselineBD, MathContext(8)).toDouble()
                    }
                }
                portfolioSummaryList.add(
                    PortfolioRow(symbolCode, currency, quantity, currentPrice, currentHoldingValue, baselineValue, deviation, absoluteDifference)
                )
            }

            if (priceFetchIssues.isNotEmpty()) {
                mainLoopLogger.warn("⚠️ Price unavailable/invalid via WebSocket for: [${priceFetchIssues.joinToString()}]. Calculations skipped for these assets.")
            }

            if (!initialized && baselinesVerifiedOrSetThisCycle) {
                mainLoopLogger.info("✅ Baselines & Timestamps init/verify complete.")
                initialized = true
            } else if (!initialized && currentHoldings.isNotEmpty() && currentHoldings.size == priceFetchIssues.size && !baselinesVerifiedOrSetThisCycle) {
                mainLoopLogger.info("⏳ Waiting for prices for baseline init (all holdings lack prices)...")
            } else if (!initialized && currentHoldings.isEmpty()) {
                mainLoopLogger.info("✅ No holdings, baseline init considered complete.")
                initialized = true
            }

            val symbolsToRemove = botState.baselines.keys.filterNot { it in currentSymbolsInPortfolio }.toSet()
            if (symbolsToRemove.isNotEmpty()) {
                symbolsToRemove.forEach { symCode ->
                    mainLoopLogger.info("🗑️ Clearing state for sold/removed asset: $symCode.")
                    botState.baselines.remove(symCode)
                    botState.trailingState.remove(symCode)
                    botState.lastActionTimestamps.remove(symCode)
                    botState.rebalanceState.remove(symCode)
                    botState.adaptiveDeadZoneState.remove(symCode) // Also clear ADZ state
                }
                stateChangedThisCycle = true
            }

            portfolioSummaryList.sortByDescending { it.deviation ?: Double.NEGATIVE_INFINITY }
            // ... (Portfolio Summary and Financial Overview logging as before) ...
             if (portfolioSummaryList.isNotEmpty()) {
                mainLoopLogger.info("--- Portfolio Summary (Sorted by Deviation %) ---")
                mainLoopLogger.info(String.format("%-10s | %-18s | %-15s | %-18s | %-12s | %-10s", "Symbol", "Quantity", "Price", "Value ($QUOTE_CURRENCY_CODE)", "Baseline", "Deviation"))
                portfolioSummaryList.forEach { row ->
                    mainLoopLogger.info(String.format("%-10s | %-18.8f | %-15s | %-18s | %-12s | %-10s",
                        row.symbol,
                        row.quantity,
                        row.price?.toPlainString() ?: "N/A",
                        row.value?.setScale(2, RoundingMode.HALF_UP)?.toPlainString() ?: "N/A",
                        row.baseline?.let { "$${"%.2f".format(it)}" } ?: "N/A",
                        row.deviation?.let { "${"%.2f".format(it * 100)}%" } ?: "N/A"
                    ))
                }
            } else if (currentHoldings.isNotEmpty() && priceFetchIssues.size == currentHoldings.size) {
                 mainLoopLogger.info("ℹ️ Portfolio summary unavailable (missing prices for all holdings).")
            } else if (currentHoldings.isEmpty()) {
                 mainLoopLogger.info("ℹ️ Portfolio empty, no summary to display.")
            }

            mainLoopLogger.info("--- Financial Overview ---")
            mainLoopLogger.info("Total Holdings Value:   $QUOTE_CURRENCY_CODE ${totalHoldingsValue.setScale(2, RoundingMode.HALF_UP).toPlainString()}")
            mainLoopLogger.info("Cash Balance:           $QUOTE_CURRENCY_CODE ${cashBalance.setScale(2, RoundingMode.HALF_UP).toPlainString()}")
            val totalPortfolioValue = totalHoldingsValue.add(cashBalance)
            mainLoopLogger.info("Total Portfolio Value:  $QUOTE_CURRENCY_CODE ${totalPortfolioValue.setScale(2, RoundingMode.HALF_UP).toPlainString()}")

            var totalBaselineDifferenceManaged = BigDecimal.ZERO
            var totalManagedBaselineValue = BigDecimal.ZERO
            var managedAssetsCount = 0
            portfolioSummaryList.forEach { row ->
                if (row.baseline != null && row.baseline > 0 && !REBALANCE_EXCLUDE_SYMBOLS.contains(row.symbol) && row.value != null && row.absoluteDifference != null) {
                    totalManagedBaselineValue = totalManagedBaselineValue.add(BigDecimal.valueOf(row.baseline))
                    totalBaselineDifferenceManaged = totalBaselineDifferenceManaged.add(row.absoluteDifference)
                    managedAssetsCount++
                }
            }
            val currentPortfolioDeviationPercentForDisplay = if (totalManagedBaselineValue > BigDecimal.ZERO) {
                totalBaselineDifferenceManaged.divide(totalManagedBaselineValue, MathContext(4)).toDouble()
            } else 0.0
            mainLoopLogger.info("Deviation (Managed):    $${totalBaselineDifferenceManaged.setScale(2, RoundingMode.HALF_UP).toPlainString()} (${"%.2f".format(currentPortfolioDeviationPercentForDisplay * 100)}%) ($managedAssetsCount Assets)")


            // --- ADZ & CP Logic ---
            if (ENABLE_ADAPTIVE_DEAD_ZONE && initialized) {
                portfolioSummaryList.forEach { row ->
                    val symCode = row.symbol
                    if (HARVEST_EXCLUDE_SYMBOLS.contains(symCode) || REBALANCE_EXCLUDE_SYMBOLS.contains(symCode)) {
                        if (botState.adaptiveDeadZoneState.remove(symCode) == true) {
                            mainLoopLogger.info("ℹ️ $symCode: Cleared adaptive DZ state (ineligible or excluded).")
                            stateChangedThisCycle = true
                        }
                        return@forEach
                    }
                    val deviation = row.deviation ?: return@forEach
                    val lastActionTime = botState.lastActionTimestamps[symCode] ?: 0L
                    val timeSinceLastAction = System.currentTimeMillis() - lastActionTime
                    val inactivityTimeoutMet = timeSinceLastAction >= ADAPTIVE_DZ_INACTIVITY_TIMEOUT
                    val isCurrentlyADZ = botState.adaptiveDeadZoneState[symCode] == true

                    val isStrictlyInOriginalDeadZone = deviation < FLAT_HARVEST_TRIGGER_PERCENT && deviation > -FLAT_REBALANCE_TRIGGER_PERCENT
                    val isOnOrOutsideOriginalDeadZone = deviation >= FLAT_HARVEST_TRIGGER_PERCENT || deviation <= -FLAT_REBALANCE_TRIGGER_PERCENT

                    if (isCurrentlyADZ && isOnOrOutsideOriginalDeadZone) {
                        botState.adaptiveDeadZoneState.remove(symCode)
                        mainLoopLogger.info("✅ $symCode: Adaptive DZ Mode DEACTIVATED (Deviation [${"%.2f".format(deviation * 100)}%] hit/exceeded original +/-${"%.1f".format(FLAT_HARVEST_TRIGGER_PERCENT*100)}% bounds).")
                        botState.trailingState[symCode]?.copy(harvestCycleCount = 0)?.also { botState.trailingState[symCode] = it }
                        botState.rebalanceState[symCode]?.copy(rebalancePosCycleCount = 0)?.also { botState.rebalanceState[symCode] = it }
                        stateChangedThisCycle = true
                    } else if (!isCurrentlyADZ && isStrictlyInOriginalDeadZone && inactivityTimeoutMet) {
                        botState.adaptiveDeadZoneState[symCode] = true
                        mainLoopLogger.info("⚡ $symCode: Adaptive DZ Mode ACTIVATED (In DZ & inactive for ${timeSinceLastAction / (60*60*1000)} hrs). Using +/-${"%.1f".format(ADAPTIVE_DZ_HARVEST_TRIGGER_PERCENT*100)}% triggers.")
                        botState.trailingState[symCode]?.copy(harvestCycleCount = 0)?.also { botState.trailingState[symCode] = it }
                        botState.rebalanceState[symCode]?.copy(rebalancePosCycleCount = 0)?.also { botState.rebalanceState[symCode] = it }
                        stateChangedThisCycle = true
                    }
                }
            }

            var isGlobalRiskSignalActive = false
            if (ENABLE_CRASH_PROTECTION && initialized) {
                val assetsWithBaseline = portfolioSummaryList.filter { it.baseline != null && it.baseline > 0.01 }
                val assetsWithBaselineCount = assetsWithBaseline.size
                if (assetsWithBaselineCount > 0) {
                    val assetsMeetingDeclineThresholdCount = assetsWithBaseline.count {
                        it.deviation != null && it.deviation <= CP_TRIGGER_MIN_NEGATIVE_DEV_PERCENT
                    }
                    val percentageMeetingThreshold = if (assetsWithBaselineCount > 0) assetsMeetingDeclineThresholdCount.toDouble() / assetsWithBaselineCount else 0.0
                    if (percentageMeetingThreshold >= CP_TRIGGER_ASSET_PERCENT) {
                        isGlobalRiskSignalActive = true
                        mainLoopLogger.info("🛡️ Crash Protection ACTIVE (${"%.1f".format(percentageMeetingThreshold * 100)}% >= ${"%.0f".format(CP_TRIGGER_ASSET_PERCENT * 100)}% of assets <= ${"%.1f".format(CP_TRIGGER_MIN_NEGATIVE_DEV_PERCENT * 100)}% dev)")
                    } else if (isGlobalRiskSignalActive) { // Was active, now not
                        mainLoopLogger.info("🛡️ Crash Protection DEACTIVATED.")
                    }
                }
            }

            // --- Start Trading Logic ---
            val validPortfolioItems = portfolioSummaryList.filter { it.baseline != null && it.baseline > 0.01 && it.deviation != null && it.price != null && it.price > BigDecimal.ZERO && it.value != null }

            if (!initialized) {
                mainLoopLogger.info("⏳ Baselines not fully initialized, skipping trading logic.")
            } else if (validPortfolioItems.isEmpty() && currentHoldings.isNotEmpty()) {
                mainLoopLogger.info("📉 No assets with valid data for decisions (e.g. missing prices), skipping trading logic.")
            } else if (currentHoldings.isEmpty() && !(HARVEST_ALLOC_BTC_PERCENT > 0 && MIN_BTC_BUY_USD < 1000 && cashBalance >= BigDecimal.valueOf(MIN_BTC_BUY_USD))) {
                mainLoopLogger.info("🧘 No holdings to manage, skipping trading logic (BTC buy not triggered or insufficient cash).")
            } else {
                mainLoopLogger.info("🚦 Baselines ready. Proceeding with trading logic...")
                var portfolioHarvestExecutedThisCycle = false

                // --- Portfolio Override Harvest Logic ---
                if (ENABLE_PORTFOLIO_HARVEST) {
                    val currentPortfolioDeviationPercent = currentPortfolioDeviationPercentForDisplay // Use the already calculated value

                    if (!portfolioHarvestState.flagged && currentPortfolioDeviationPercent >= PORTFOLIO_HARVEST_TRIGGER_DEVIATION_PERCENT) {
                        portfolioHarvestState = PortfolioHarvestStateData(true, 0, System.currentTimeMillis(), currentPortfolioDeviationPercent)
                        mainLoopLogger.info("📈 Portfolio flagged for Baseline Reset Harvest at ${"%.2f".format(currentPortfolioDeviationPercent * 100)}% (>= ${"%.2f".format(PORTFOLIO_HARVEST_TRIGGER_DEVIATION_PERCENT * 100)}%).")
                    } else if (portfolioHarvestState.flagged && currentPortfolioDeviationPercent < PORTFOLIO_HARVEST_TRIGGER_DEVIATION_PERCENT) {
                        mainLoopLogger.info("📉 Portfolio dropped below Baseline Reset Harvest trigger. Clearing flag.")
                        portfolioHarvestState = PortfolioHarvestStateData()
                    }

                    if (portfolioHarvestState.flagged) {
                        val prevDev = portfolioHarvestState.previousDeviationPercent
                        if (prevDev != null) {
                            if (currentPortfolioDeviationPercent < prevDev) {
                                portfolioHarvestState.cycleCount++
                                mainLoopLogger.info("📊 P-Harvest: Dev decreased. Count INC to ${portfolioHarvestState.cycleCount}.")
                            } else if (currentPortfolioDeviationPercent > prevDev) {
                                portfolioHarvestState.cycleCount = max(0, portfolioHarvestState.cycleCount - 1)
                                mainLoopLogger.info("📊 P-Harvest: Dev increased. Count DEC to ${portfolioHarvestState.cycleCount}.")
                            }
                        }
                        portfolioHarvestState.previousDeviationPercent = currentPortfolioDeviationPercent
                    }

                    if (portfolioHarvestState.flagged && portfolioHarvestState.cycleCount >= PORTFOLIO_HARVEST_CONFIRMATION_CYCLES) {
                        mainLoopLogger.info("🎉 Executing Portfolio Baseline Reset Harvest!")
                        portfolioHarvestExecutedThisCycle = true
                        var totalHarvestedThisEvent = BigDecimal.ZERO
                        var assetsSoldCount = 0
                        val assetsToUpdateTimestamp = mutableListOf<String>()

                        validPortfolioItems.forEach { row ->
                            if (REBALANCE_EXCLUDE_SYMBOLS.contains(row.symbol) || HARVEST_EXCLUDE_SYMBOLS.contains(row.symbol) || row.value == null || row.baseline == null || row.value <= BigDecimal.valueOf(row.baseline) || row.price == null) return@forEach
                            val originalBaseline = row.baseline
                            val surplusUSD = row.value.subtract(BigDecimal.valueOf(originalBaseline))
                            if (surplusUSD < BigDecimal.valueOf(MIN_ASSET_SURPLUS_FOR_PORTFOLIO_HARVEST)) return@forEach

                            val qtyToSell = surplusUSD.divide(row.price, MathContext.DECIMAL64)
                            val roundedQtyToSell = roundQuantity(row.currency, CurrencyPair(row.currency, QUOTE_CURRENCY), qtyToSell)

                            if (roundedQtyToSell > BigDecimal.ZERO) {
                                assetsSoldCount++
                                mainLoopLogger.info("   -> Selling P-Harvest surplus $roundedQtyToSell ${row.symbol} (~$${surplusUSD.setScale(2, RoundingMode.HALF_UP).toPlainString()})")
                                val sellResp = ExchangeService.placeMarketOrder(CurrencyPair(row.currency, QUOTE_CURRENCY), Order.OrderType.ASK, OrderAmount.BaseSize(roundedQtyToSell))
                                if (sellResp != null) {
                                    val actualSoldValue = roundedQtyToSell.multiply(row.price)
                                    mainLoopLogger.info("   ✅ ${row.symbol}: Sold ~$${actualSoldValue.setScale(2, RoundingMode.HALF_UP).toPlainString()}. ID: $sellResp")
                                    logTrade(row.symbol, "SELL", roundedQtyToSell.toPlainString(), row.price.toPlainString(), sellResp, "Portfolio Baseline Reset Harvest")
                                    botState.baselines[row.symbol] = originalBaseline
                                    mainLoopLogger.info("   🔄 ${row.symbol}: Baseline RESET to $$originalBaseline.")
                                    assetsToUpdateTimestamp.add(row.symbol)
                                    totalHarvestedThisEvent = totalHarvestedThisEvent.add(actualSoldValue)
                                    botState.trailingState.remove(row.symbol)
                                } else {
                                    mainLoopLogger.warn("   ⚠️ ${row.symbol}: P-Harvest sell FAILED or NO ID. Baseline NOT reset.")
                                }
                            }
                        }

                        if (assetsSoldCount > 0) {
                            harvestedAmountThisCycle = harvestedAmountThisCycle.add(totalHarvestedThisEvent)
                            anyTradesThisCycle = true
                            stateChangedThisCycle = true
                            assetsToUpdateTimestamp.forEach { sym ->
                                botState.lastActionTimestamps[sym] = System.currentTimeMillis()
                                mainLoopLogger.info("   ⏱️ $sym: Updated last action timestamp (Portfolio Harvest).")
                            }
                        }
                        mainLoopLogger.info("🏁 P-Harvest finished. Total ~$${totalHarvestedThisEvent.setScale(2,RoundingMode.HALF_UP).toPlainString()} from $assetsSoldCount assets.")
                        portfolioHarvestState = PortfolioHarvestStateData()
                    }
                }

                // --- Individual Asset Harvest Logic ---
                if (!portfolioHarvestExecutedThisCycle) {
                    validPortfolioItems.forEach { row ->
                        val symCode = row.symbol
                        val currency = row.currency
                        val pair = CurrencyPair(currency, QUOTE_CURRENCY)
                        val currentBaseline = row.baseline ?: return@forEach
                        if (HARVEST_EXCLUDE_SYMBOLS.contains(symCode)) return@forEach

                        val currentPrice = row.price ?: return@forEach
                        val currentValue = row.value ?: return@forEach
                        val currentDeviation = row.deviation ?: return@forEach

                        val isADZActiveForSymbol = botState.adaptiveDeadZoneState[symCode] == true
                        val effectiveHarvestTriggerPercent = if (isADZActiveForSymbol) ADAPTIVE_DZ_HARVEST_TRIGGER_PERCENT else FLAT_HARVEST_TRIGGER_PERCENT
                        val requiredHarvestCycles = if (isADZActiveForSymbol) HARVEST_CYCLE_THRESHOLD + 1 else HARVEST_CYCLE_THRESHOLD

                        val pairMeta = assetPairMetaData[pair]
                        val minOrderQtyFromMeta = pairMeta?.minimumAmount ?: BigDecimal.ZERO
                        val minSellValue = if (minOrderQtyFromMeta > BigDecimal.ZERO && currentPrice > BigDecimal.ZERO) minOrderQtyFromMeta.multiply(currentPrice) else BigDecimal.ZERO

                        val upperBandValue = BigDecimal.valueOf(currentBaseline * (1 + effectiveHarvestTriggerPercent))

                        var st = botState.trailingState[symCode]
                        if (st == null) st = TrailingData() // No need to put in map yet

                        if (!st.flagged && currentValue >= upperBandValue) {
                            botState.trailingState[symCode] = st.copy(flagged = true, harvestCycleCount = 0, flaggedAt = System.currentTimeMillis(), previousDeviation = currentDeviation)
                            mainLoopLogger.info("🚩 $symCode flagged for Harvest at $${currentValue.setScale(2,RoundingMode.HALF_UP).toPlainString()} (Dev: ${"%.2f".format(currentDeviation * 100)}% >= ${"%.2f".format(effectiveHarvestTriggerPercent*100)}%). ADZ: $isADZActiveForSymbol")
                            stateChangedThisCycle = true
                            st = botState.trailingState[symCode]!!
                        } else if (st.flagged && currentValue < upperBandValue) {
                            mainLoopLogger.info("📉 $symCode dropped below Harvest trigger ($${upperBandValue.setScale(2,RoundingMode.HALF_UP).toPlainString()}). Clearing flag. ADZ: $isADZActiveForSymbol")
                            botState.trailingState.remove(symCode)
                            stateChangedThisCycle = true
                            return@forEach
                        }

                        if (st.flagged == false) return@forEach

                        val flaggedAtTime = st.flaggedAt ?: System.currentTimeMillis()
                        val flaggedDuration = System.currentTimeMillis() - flaggedAtTime

                        if (flaggedDuration > FORCED_HARVEST_TIMEOUT) {
                            val surplus = currentValue.subtract(BigDecimal.valueOf(currentBaseline))
                            if (surplus < BigDecimal.valueOf(MIN_SURPLUS_FOR_FORCED_HARVEST) || (minSellValue > BigDecimal.ZERO && surplus < minSellValue)) {
                                val reason = if (surplus < BigDecimal.valueOf(MIN_SURPLUS_FOR_FORCED_HARVEST)) "Surplus $${surplus.setScale(2,RoundingMode.HALF_UP).toPlainString()} < min $MIN_SURPLUS_FOR_FORCED_HARVEST" else "Surplus $${surplus.setScale(2,RoundingMode.HALF_UP).toPlainString()} < min order value $${minSellValue.setScale(2,RoundingMode.HALF_UP).toPlainString()}"
                                mainLoopLogger.info("ℹ️ $symCode (Forced Harvest): $reason. Clearing flag.")
                                botState.trailingState.remove(symCode)
                                stateChangedThisCycle = true
                                return@forEach
                            }
                            val qtyToSell = surplus.divide(currentPrice, MathContext.DECIMAL64)
                            val roundedQtyToSell = roundQuantity(currency, pair, qtyToSell)

                            if (roundedQtyToSell > BigDecimal.ZERO) {
                                mainLoopLogger.info("⏳ Attempting Forced Harvest $symCode: Selling $roundedQtyToSell (~$${surplus.setScale(2,RoundingMode.HALF_UP).toPlainString()}) due to timeout.")
                                val orderId = ExchangeService.placeMarketOrder(pair, Order.OrderType.ASK, OrderAmount.BaseSize(roundedQtyToSell))
                                if (orderId != null) {
                                    val sellValue = roundedQtyToSell.multiply(currentPrice)
                                    mainLoopLogger.info("✅ (Forced Harvest) $symCode: Sold ~$${sellValue.setScale(2,RoundingMode.HALF_UP).toPlainString()}. ID: $orderId")
                                    logTrade(symCode, "SELL", roundedQtyToSell.toPlainString(), currentPrice.toPlainString(), orderId, "Forced Harvest (Timeout)")
                                    harvestedAmountThisCycle = harvestedAmountThisCycle.add(sellValue)
                                    anyTradesThisCycle = true
                                    botState.baselines[symCode] = currentBaseline * (1 + TARGET_ADJUST_PERCENT)
                                    mainLoopLogger.info("📈 $symCode: Baseline adjusted to $${botState.baselines[symCode]?.let{"%.2f".format(it)}} (Forced Harvest).")
                                    botState.lastActionTimestamps[symCode] = System.currentTimeMillis()
                                    mainLoopLogger.info("⏱️ $symCode: Updated last action timestamp (Forced Harvest).")
                                    botState.trailingState.remove(symCode)
                                    stateChangedThisCycle = true
                                } else {
                                     mainLoopLogger.warn("⚠️ Forced Harvest $symCode: sell order FAILED or no ID. Clearing flag.")
                                     botState.trailingState.remove(symCode)
                                     stateChangedThisCycle = true
                                }
                            } else {
                                mainLoopLogger.info("ℹ️ $symCode (Forced Harvest): Rounded Qty '$roundedQtyToSell' too small. Clearing flag.")
                                botState.trailingState.remove(symCode)
                                stateChangedThisCycle = true
                            }
                            return@forEach
                        }

                        var currentTrailingData = botState.trailingState[symCode] ?: st // Use 'st' if not updated by remove
                        val prevDeviationForCycle = currentTrailingData.previousDeviation
                        if (prevDeviationForCycle != null) {
                            if (currentDeviation < prevDeviationForCycle) {
                                currentTrailingData = currentTrailingData.copy(harvestCycleCount = currentTrailingData.harvestCycleCount + 1)
                                mainLoopLogger.info("📊 $symCode Harvest: Dev decreased (${"%.2f".format(prevDeviationForCycle * 100)}% -> ${"%.2f".format(currentDeviation * 100)}%). Count INC to ${currentTrailingData.harvestCycleCount}. ADZ: $isADZActiveForSymbol")
                            } else if (currentDeviation > prevDeviationForCycle) {
                                currentTrailingData = currentTrailingData.copy(harvestCycleCount = max(0, currentTrailingData.harvestCycleCount - 1))
                                 mainLoopLogger.info("📊 $symCode Harvest: Dev increased (${"%.2f".format(prevDeviationForCycle * 100)}% -> ${"%.2f".format(currentDeviation * 100)}%). Count DEC to ${currentTrailingData.harvestCycleCount}. ADZ: $isADZActiveForSymbol")
                            }
                        } else { // First cycle after flagging
                             currentTrailingData = currentTrailingData.copy(harvestCycleCount = 0)
                        }
                        if (botState.trailingState[symCode] != currentTrailingData.copy(previousDeviation = currentDeviation)) { // Only update if actual change
                            botState.trailingState[symCode] = currentTrailingData.copy(previousDeviation = currentDeviation)
                            stateChangedThisCycle = true
                        }

                        if (currentTrailingData.harvestCycleCount >= requiredHarvestCycles) {
                            val surplus = currentValue.subtract(BigDecimal.valueOf(currentBaseline))
                             if (surplus < BigDecimal.valueOf(MIN_SURPLUS_FOR_HARVEST) || (minSellValue > BigDecimal.ZERO && surplus < minSellValue)) {
                                val reason = if (surplus < BigDecimal.valueOf(MIN_SURPLUS_FOR_HARVEST)) "Surplus $${surplus.setScale(2,RoundingMode.HALF_UP).toPlainString()} < min $MIN_SURPLUS_FOR_HARVEST" else "Surplus $${surplus.setScale(2,RoundingMode.HALF_UP).toPlainString()} < min order value $${minSellValue.setScale(2,RoundingMode.HALF_UP).toPlainString()}"
                                mainLoopLogger.info("ℹ️ $symCode (Harvest): $reason. Resetting count. ADZ: $isADZActiveForSymbol")
                                botState.trailingState[symCode] = currentTrailingData.copy(harvestCycleCount = 0, previousDeviation = null)
                                return@forEach
                            }
                            val qtyToSell = surplus.divide(currentPrice, MathContext.DECIMAL64)
                            val roundedQtyToSell = roundQuantity(currency, pair, qtyToSell)

                            if (roundedQtyToSell > BigDecimal.ZERO) {
                                mainLoopLogger.info("📉 Attempting Standard Harvest $symCode: Selling $roundedQtyToSell (~$${surplus.setScale(2,RoundingMode.HALF_UP).toPlainString()}) ($requiredHarvestCycles cycles). ADZ: $isADZActiveForSymbol")
                                val orderId = ExchangeService.placeMarketOrder(pair, Order.OrderType.ASK, OrderAmount.BaseSize(roundedQtyToSell))
                                if (orderId != null) {
                                    val sellValue = roundedQtyToSell.multiply(currentPrice)
                                    mainLoopLogger.info("✅ Harvest $symCode: Sold ~$${sellValue.setScale(2,RoundingMode.HALF_UP).toPlainString()}. ID: $orderId")
                                    logTrade(symCode, "SELL", roundedQtyToSell.toPlainString(), currentPrice.toPlainString(), orderId, "Harvest ($requiredHarvestCycles cycles, ADZ: $isADZActiveForSymbol)")
                                    harvestedAmountThisCycle = harvestedAmountThisCycle.add(sellValue)
                                    anyTradesThisCycle = true
                                    botState.baselines[symCode] = currentBaseline * (1 + TARGET_ADJUST_PERCENT)
                                    mainLoopLogger.info("📈 $symCode: Baseline adjusted to $${botState.baselines[symCode]?.let{"%.2f".format(it)}} (Harvest).")
                                    botState.lastActionTimestamps[symCode] = System.currentTimeMillis()
                                    mainLoopLogger.info("⏱️ $symCode: Updated last action timestamp (Harvest).")
                                    botState.trailingState.remove(symCode)
                                    stateChangedThisCycle = true
                                } else {
                                    mainLoopLogger.warn("⚠️ Harvest $symCode: sell order FAILED or no ID. Resetting count. ADZ: $isADZActiveForSymbol")
                                    botState.trailingState[symCode] = currentTrailingData.copy(harvestCycleCount = 0, previousDeviation = null)
                                }
                            } else {
                                 mainLoopLogger.info("ℹ️ $symCode (Harvest): Rounded Qty '$roundedQtyToSell' too small. Resetting count. ADZ: $isADZActiveForSymbol")
                                 botState.trailingState[symCode] = currentTrailingData.copy(harvestCycleCount = 0, previousDeviation = null)
                            }
                        }
                    }
                }
            }

            // --- Harvest Proceeds Allocation ---
            var totalReinvestedThisCycle = BigDecimal.ZERO
            if (harvestedAmountThisCycle >= BigDecimal.valueOf(MIN_HARVEST_TO_ALLOCATE)) {
                var amountForReinvest = harvestedAmountThisCycle.multiply(BigDecimal.valueOf(HARVEST_ALLOC_REINVEST_PERCENT))
                val amountForBTC = harvestedAmountThisCycle.multiply(BigDecimal.valueOf(HARVEST_ALLOC_BTC_PERCENT))
                mainLoopLogger.info("💵 Harvest Allocation: Total $${harvestedAmountThisCycle.setScale(2,RoundingMode.HALF_UP).toPlainString()} -> Reinvest: $${amountForReinvest.setScale(2,RoundingMode.HALF_UP).toPlainString()}, BTC: $${amountForBTC.setScale(2,RoundingMode.HALF_UP).toPlainString()}")

                if (amountForReinvest > BigDecimal.ZERO) {
                    val reinvestmentCandidates = validPortfolioItems // Use already filtered list
                        .filter { row ->
                            !REBALANCE_EXCLUDE_SYMBOLS.contains(row.symbol) &&
                            row.baseline != null && row.baseline > 0 &&
                            row.value != null && row.price != null && row.price > BigDecimal.ZERO &&
                            row.value < BigDecimal.valueOf(row.baseline) &&
                            row.deviation != null && row.deviation <= MIN_NEGATIVE_DEVIATION_FOR_REINVEST
                        }
                        .sortedBy { it.deviation }

                    if (reinvestmentCandidates.isNotEmpty()) {
                        mainLoopLogger.info("💡 Found ${reinvestmentCandidates.size} candidate(s) for priority reinvestment (Dev <= ${MIN_NEGATIVE_DEVIATION_FOR_REINVEST * 100}%).")
                        var remainingReinvestAllocation = amountForReinvest

                        for (candidate in reinvestmentCandidates) {
                            if (remainingReinvestAllocation < BigDecimal.valueOf(MIN_REINVEST_BUY_USD)) break
                            val symCode = candidate.symbol
                            val currency = candidate.currency
                            val pair = CurrencyPair(currency, QUOTE_CURRENCY)
                            val currentBaseline = candidate.baseline!!
                            val price = candidate.price!!
                            val currentValue = candidate.value!!

                            val amountNeededToBaseline = BigDecimal.valueOf(currentBaseline).subtract(currentValue).max(BigDecimal.ZERO)
                            if (amountNeededToBaseline <= BigDecimal.valueOf(0.01)) continue

                            var buyAmountUSD = amountNeededToBaseline.min(remainingReinvestAllocation)
                            if (buyAmountUSD < BigDecimal.valueOf(MIN_REINVEST_BUY_USD)) continue

                            var qtyToBuy = buyAmountUSD.divide(price, MathContext.DECIMAL64)
                            val pairMeta = assetPairMetaData[pair]
                            val minOrderQtyFromMeta = pairMeta?.minimumAmount ?: BigDecimal.ZERO
                            if (minOrderQtyFromMeta > BigDecimal.ZERO && qtyToBuy > BigDecimal.ZERO && qtyToBuy < minOrderQtyFromMeta) {
                                if (minOrderQtyFromMeta.multiply(price) <= remainingReinvestAllocation && minOrderQtyFromMeta.multiply(price) >= BigDecimal.valueOf(MIN_REINVEST_BUY_USD) ) {
                                    qtyToBuy = minOrderQtyFromMeta
                                    buyAmountUSD = qtyToBuy.multiply(price) // Recalculate actual USD amount for this min qty
                                    mainLoopLogger.info("   ℹ️ $symCode (Priority Reinvest): Overriding to min qty $qtyToBuy (~$${buyAmountUSD.setScale(2,RoundingMode.HALF_UP).toPlainString()}).")
                                } else {
                                    mainLoopLogger.info("   ℹ️ $symCode (Priority Reinvest): Desired qty $qtyToBuy or min qty $minOrderQtyFromMeta too small/costly for budget ($remainingReinvestAllocation). Skipping.")
                                    continue
                                }
                            }
                            val roundedQtyToBuy = roundQuantity(currency, pair, qtyToBuy)

                            if (roundedQtyToBuy > BigDecimal.ZERO) {
                                mainLoopLogger.info("    R🛒 Attempting Priority Reinvestment $symCode: Buying $roundedQtyToBuy (~$${buyAmountUSD.setScale(2,RoundingMode.HALF_UP).toPlainString()}) to reach baseline $$currentBaseline.")
                                val orderId = ExchangeService.placeMarketOrder(pair, Order.OrderType.BID, OrderAmount.QuoteSize(buyAmountUSD))
                                if (orderId != null) {
                                    val actualCost = roundedQtyToBuy.multiply(price)
                                    mainLoopLogger.info("   ✅ Reinvest $symCode: Spent ~$${actualCost.setScale(2,RoundingMode.HALF_UP).toPlainString()}. ID: $orderId")
                                    logTrade(symCode, "BUY", roundedQtyToBuy.toPlainString(), price.toPlainString(), orderId, "Priority Reinvestment Buy (from harvest)")
                                    totalReinvestedThisCycle = totalReinvestedThisCycle.add(actualCost)
                                    remainingReinvestAllocation = remainingReinvestAllocation.subtract(actualCost)
                                    anyTradesThisCycle = true
                                    botState.lastActionTimestamps[symCode] = System.currentTimeMillis()
                                    mainLoopLogger.info("   ⏱️ $symCode: Updated last action timestamp (Priority Reinvest).")
                                    if (botState.rebalanceState.containsKey(symCode)) {
                                        mainLoopLogger.info("   🗑️ Clearing standard rebalance state for $symCode after priority reinvestment.")
                                        botState.rebalanceState.remove(symCode)
                                    }
                                    stateChangedThisCycle = true
                                } else {
                                     mainLoopLogger.warn("   ⚠️ Priority Reinvest $symCode: Buy order FAILED or no ID.")
                                }
                            }
                        }
                        mainLoopLogger.info("🏁 Priority Reinvestment finished. Total spent: ~$${totalReinvestedThisCycle.setScale(2,RoundingMode.HALF_UP).toPlainString()} / $${amountForReinvest.setScale(2,RoundingMode.HALF_UP).toPlainString()} allocated.")
                    } else {
                        mainLoopLogger.info("ℹ️ No candidates met priority reinvestment criteria.")
                    }
                }

                val amountToCashCalculated = harvestedAmountThisCycle.subtract(totalReinvestedThisCycle).subtract(amountForBTC)
                mainLoopLogger.info("   -> Final Allocation: Reinvested $${totalReinvestedThisCycle.setScale(2,RoundingMode.HALF_UP).toPlainString()}, To BTC $${amountForBTC.setScale(2,RoundingMode.HALF_UP).toPlainString()}, To Cash $${amountToCashCalculated.setScale(2,RoundingMode.HALF_UP).toPlainString()}")

                if (amountForBTC >= BigDecimal.valueOf(MIN_BTC_BUY_USD) && cashBalance >= amountForBTC) {
                    mainLoopLogger.info("₿ Attempting Auto BTC Buy from Harvest: $${amountForBTC.setScale(2,RoundingMode.HALF_UP).toPlainString()}")
                    val btcPair = CurrencyPair.BTC_USD
                    val btcPriceTicker = latestPrices[btcPair]
                    if (btcPriceTicker?.last != null && btcPriceTicker.last > BigDecimal.ZERO) {
                        // val btcQtyToBuy = amountForBTC.divide(btcPriceTicker.last, MathContext.DECIMAL64) // This would be base size
                        // val roundedBtcQty = roundQuantity(Currency.BTC, btcPair, btcQtyToBuy)
                        // For QuoteSize order, the amount IS the quote currency amount
                        val orderId = ExchangeService.placeMarketOrder(btcPair, Order.OrderType.BID, OrderAmount.QuoteSize(amountForBTC))
                        if (orderId != null) {
                            // Log actual quantity if available from order response, for now log the quote amount
                            logTrade("BTC", "BUY", amountForBTC.setScale(2,RoundingMode.HALF_UP).toPlainString() + " " + QUOTE_CURRENCY_CODE, btcPriceTicker.last.toPlainString(), orderId, "Harvest Allocation BTC Buy")
                            anyTradesThisCycle = true
                            botState.lastActionTimestamps[Currency.BTC.currencyCode] = System.currentTimeMillis()
                            stateChangedThisCycle = true
                        } else {
                            mainLoopLogger.warn("⚠️ Auto BTC Buy from Harvest FAILED.")
                        }
                    } else {
                        mainLoopLogger.warn("⚠️ Auto BTC Buy: Could not get BTC price or price is zero.")
                    }
                } else if (harvestedAmountThisCycle >= BigDecimal.valueOf(MIN_HARVEST_TO_ALLOCATE) && HARVEST_ALLOC_BTC_PERCENT > 0 && amountForBTC > BigDecimal.ZERO) {
                    mainLoopLogger.info("ℹ️ BTC allocation $${amountForBTC.setScale(2,RoundingMode.HALF_UP).toPlainString()} is less than minimum buy $${MIN_BTC_BUY_USD} or insufficient cash ($${cashBalance.setScale(2,RoundingMode.HALF_UP).toPlainString()}). Skipping BTC buy.")
                }

            } else if (harvestedAmountThisCycle > BigDecimal.ZERO) {
                mainLoopLogger.info("💵 Harvested $${harvestedAmountThisCycle.setScale(2,RoundingMode.HALF_UP).toPlainString()}, below minimum to allocate ($MIN_HARVEST_TO_ALLOCATE). Treating as cash.")
            }

            // --- Rebalancing Logic (Standard) ---
            if (!portfolioHarvestExecutedThisCycle) {
                validPortfolioItems.forEach { row ->
                    val symCode = row.symbol
                    val currency = row.currency
                    val pair = CurrencyPair(currency, QUOTE_CURRENCY)
                    val currentBaseline = row.baseline ?: return@forEach
                    val isADZActiveForSymbol = botState.adaptiveDeadZoneState[symCode] == true

                    if (REBALANCE_EXCLUDE_SYMBOLS.contains(symCode) || (botState.trailingState[symCode]?.flagged == true) ) {
                        if(botState.rebalanceState.containsKey(symCode)) {
                            mainLoopLogger.debug("Clearing rebalance state for $symCode due to exclusion or harvest flag.")
                            botState.rebalanceState.remove(symCode)
                            stateChangedThisCycle = true
                        }
                        return@forEach
                    }
                    val currentValue = row.value ?: return@forEach
                    val currentPrice = row.price ?: return@forEach
                    val currentDeviation = row.deviation ?: return@forEach

                    val effectiveRebalanceTriggerPercent = if (isADZActiveForSymbol) ADAPTIVE_DZ_REBALANCE_TRIGGER_PERCENT else FLAT_REBALANCE_TRIGGER_PERCENT
                    val lowerBandValue = BigDecimal.valueOf(currentBaseline * (1 - effectiveRebalanceTriggerPercent))

                    if (currentValue >= lowerBandValue) {
                        if (botState.rebalanceState.containsKey(symCode)) {
                            mainLoopLogger.info("📈 $symCode: Value recovered above rebalance trigger ($${lowerBandValue.setScale(2,RoundingMode.HALF_UP).toPlainString()}). Clearing rebalance state. ADZ: $isADZActiveForSymbol")
                            botState.rebalanceState.remove(symCode)
                            stateChangedThisCycle = true
                        }
                        return@forEach
                    }

                    var rSt = botState.rebalanceState[symCode]
                    if (rSt == null) {
                        rSt = RebalanceData(triggered = true, triggeredAt = System.currentTimeMillis(), currentBaselineWhenTriggered = currentBaseline, previousDeviation = currentDeviation)
                        botState.rebalanceState[symCode] = rSt
                        mainLoopLogger.info("⚖️ $symCode: Rebalance triggered at $${currentValue.setScale(2,RoundingMode.HALF_UP).toPlainString()} (Dev: ${"%.2f".format(currentDeviation * 100)}% <= -${"%.2f".format(effectiveRebalanceTriggerPercent*100)}%). ADZ: $isADZActiveForSymbol")
                        stateChangedThisCycle = true
                    } else {
                         if (rSt.previousDeviation != currentDeviation) {
                            rSt = rSt.copy(previousDeviation = currentDeviation)
                            // botState.rebalanceState[symCode] = rSt // Will be saved if cycle count changes too
                            // stateChangedThisCycle = true // Only set if actual state change occurs
                        }
                    }

                    val rebalanceActiveDuration = System.currentTimeMillis() - (rSt.triggeredAt ?: System.currentTimeMillis())
                    if (rSt.triggered && rebalanceActiveDuration > FORCE_REBALANCE_TIMEOUT && rSt.attemptCount < MAX_REBALANCE_ATTEMPTS) {
                        val shortfallFromBaseline = BigDecimal.valueOf(rSt.currentBaselineWhenTriggered!!).subtract(currentValue)
                        var targetRecoveryAmount = shortfallFromBaseline.multiply(BigDecimal.valueOf(FORCE_REBALANCE_SHORTFALL_PERCENT))
                        targetRecoveryAmount = targetRecoveryAmount.max(BigDecimal.valueOf(MIN_FORCED_REBALANCE_USD))


                        if (cashBalance >= targetRecoveryAmount) {
                            mainLoopLogger.info("⏳ Attempting Forced Rebalance for $symCode: Target BUY ~$${targetRecoveryAmount.setScale(2,RoundingMode.HALF_UP).toPlainString()} due to timeout.")
                            // val qtyToBuy = targetRecoveryAmount.divide(currentPrice, MathContext.DECIMAL64) // This is quote amount
                            // val roundedQtyToBuy = roundQuantity(currency, pair, qtyToBuy) // Not needed if using QuoteSize
                            // For QuoteSize order, the amount IS the quote currency amount
                            val orderId = ExchangeService.placeMarketOrder(pair, Order.OrderType.BID, OrderAmount.QuoteSize(targetRecoveryAmount))
                            if (orderId != null) {
                                 logTrade(symCode, "BUY", targetRecoveryAmount.setScale(2, RoundingMode.HALF_UP).toPlainString() + " " + QUOTE_CURRENCY_CODE, currentPrice.toPlainString(), orderId, "Forced Rebalance (Timeout)")
                                 anyTradesThisCycle = true
                                 botState.lastActionTimestamps[symCode] = System.currentTimeMillis()
                                 // Baseline NOT adjusted for forced rebalance in JS, it aims to bring it closer to original baseline.
                                 botState.rebalanceState[symCode] = rSt.copy(attemptCount = rSt.attemptCount + 1, cooldownUntil = System.currentTimeMillis() + REBALANCE_COOLDOWN, rebalancePosCycleCount = 0, previousDeviation = null)
                                 mainLoopLogger.info("   ✅ Forced Rebalance $symCode: BUY executed. Attempt ${rSt.attemptCount + 1}. Cooldown for ${REBALANCE_COOLDOWN/60000}m.")
                                 stateChangedThisCycle = true
                            } else {
                                mainLoopLogger.warn("   ⚠️ Forced Rebalance $symCode: BUY order FAILED. Attempt ${rSt.attemptCount + 1}. Cooldown initiated.")
                                botState.rebalanceState[symCode] = rSt.copy(attemptCount = rSt.attemptCount + 1, cooldownUntil = System.currentTimeMillis() + REBALANCE_COOLDOWN)
                                stateChangedThisCycle = true
                            }
                        } else {
                            mainLoopLogger.info("   ℹ️ Forced Rebalance $symCode: Target recovery $${targetRecoveryAmount.setScale(2,RoundingMode.HALF_UP).toPlainString()} too small or insufficient cash $${cashBalance.setScale(2,RoundingMode.HALF_UP).toPlainString()}. Incrementing attempt.")
                            botState.rebalanceState[symCode] = rSt.copy(attemptCount = rSt.attemptCount + 1, cooldownUntil = System.currentTimeMillis() + REBALANCE_COOLDOWN)
                            stateChangedThisCycle = true
                        }
                        return@forEach
                    } else if (rSt.triggered && rebalanceActiveDuration > FORCE_REBALANCE_TIMEOUT && rSt.attemptCount >= MAX_REBALANCE_ATTEMPTS) {
                         mainLoopLogger.warn("⚠️ $symCode: Max forced rebalance attempts ($MAX_REBALANCE_ATTEMPTS) reached. Clearing rebalance flag. Manual review needed.")
                         botState.rebalanceState.remove(symCode)
                         stateChangedThisCycle = true
                         return@forEach
                    }

                    if (System.currentTimeMillis() < rSt.cooldownUntil) {
                        mainLoopLogger.debug("$symCode is in rebalance cooldown until ${Instant.ofEpochMilli(rSt.cooldownUntil)}. Skipping standard rebalance check.")
                        return@forEach
                    }

                    var rStMutable = rSt // Create a mutable copy for this cycle's logic
                    val prevRebalanceDev = rStMutable.previousDeviation
                    if (prevRebalanceDev != null) {
                        if (currentDeviation > prevRebalanceDev) {
                            rStMutable = rStMutable.copy(rebalancePosCycleCount = max(0, rStMutable.rebalancePosCycleCount -1))
                            mainLoopLogger.info("📊 $symCode rebalance: Dev worsened (${"%.2f".format(prevRebalanceDev * 100)}% -> ${"%.2f".format(currentDeviation * 100)}%). PosCount DEC to ${rStMutable.rebalancePosCycleCount}. ADZ: $isADZActiveForSymbol")
                        } else if (currentDeviation < prevRebalanceDev) {
                             rStMutable = rStMutable.copy(rebalancePosCycleCount = rStMutable.rebalancePosCycleCount + 1)
                             mainLoopLogger.info("📊 $symCode rebalance: Dev improved (${"%.2f".format(prevRebalanceDev * 100)}% -> ${"%.2f".format(currentDeviation * 100)}%). PosCount INC to ${rStMutable.rebalancePosCycleCount}. ADZ: $isADZActiveForSymbol")
                        }
                    }
                    // Update previousDeviation for the next cycle, regardless of posCycleCount change
                    if (rStMutable.previousDeviation != currentDeviation) {
                        rStMutable = rStMutable.copy(previousDeviation = currentDeviation)
                    }


                    val baseEffectiveRebalanceThreshold = REBALANCE_POSITIVE_THRESHOLD + (if (isGlobalRiskSignalActive) CRASH_PROTECTION_THRESHOLD_INCREASE else 0)
                    val requiredRebalanceCycles = if (isADZActiveForSymbol) baseEffectiveRebalanceThreshold + 1 else baseEffectiveRebalanceThreshold
                    val effectivePartialRecoveryPercent = if (isGlobalRiskSignalActive) PARTIAL_RECOVERY_PERCENT * CRASH_PROTECTION_PARTIAL_RECOVERY_PERCENT_FACTOR else PARTIAL_RECOVERY_PERCENT

                    if (rStMutable.rebalancePosCycleCount >= requiredRebalanceCycles && rStMutable.attemptCount < MAX_REBALANCE_ATTEMPTS) {
                        val amountToRecoverQuote = BigDecimal.valueOf(rStMutable.currentBaselineWhenTriggered!! * effectivePartialRecoveryPercent).subtract(currentValue)

                        if (amountToRecoverQuote >= BigDecimal.valueOf(MIN_PARTIAL_REBALANCE_USD) && cashBalance >= amountToRecoverQuote) {
                            mainLoopLogger.info("📈 Attempting Standard Partial Rebalance for $symCode: Target BUY ~$${amountToRecoverQuote.setScale(2,RoundingMode.HALF_UP).toPlainString()}")
                            // val qtyToBuy = amountToRecoverQuote.divide(currentPrice, MathContext.DECIMAL64) // This is quote amount
                            // val roundedQtyToBuy = roundQuantity(currency, pair, qtyToBuy) // Not needed if using QuoteSize
                            val orderId = ExchangeService.placeMarketOrder(pair, Order.OrderType.BID, OrderAmount.QuoteSize(amountToRecoverQuote))
                            if (orderId != null) {
                                logTrade(symCode, "BUY", amountToRecoverQuote.setScale(2, RoundingMode.HALF_UP).toPlainString() + " " + QUOTE_CURRENCY_CODE, currentPrice.toPlainString(), orderId, "Standard Rebalance (Partial)")
                                anyTradesThisCycle = true
                                botState.lastActionTimestamps[symCode] = System.currentTimeMillis()
                                rStMutable = rStMutable.copy(
                                    attemptCount = rStMutable.attemptCount + 1,
                                    cooldownUntil = System.currentTimeMillis() + REBALANCE_COOLDOWN,
                                    rebalancePosCycleCount = 0,
                                    previousDeviation = null
                                )
                                mainLoopLogger.info("   ✅ Standard Rebalance $symCode: BUY executed. Attempt ${rStMutable.attemptCount}. Cooldown for ${REBALANCE_COOLDOWN/60000}m.")
                            } else {
                                mainLoopLogger.warn("   ⚠️ Standard Rebalance $symCode: BUY order FAILED. Attempt ${rStMutable.attemptCount + 1}.")
                                rStMutable = rStMutable.copy(attemptCount = rStMutable.attemptCount + 1, cooldownUntil = System.currentTimeMillis() + REBALANCE_COOLDOWN)
                            }
                        } else {
                             mainLoopLogger.info("   ℹ️ Standard Rebalance $symCode: Target recovery $${amountToRecoverQuote.setScale(2,RoundingMode.HALF_UP).toPlainString()} too small or insufficient cash $${cashBalance.setScale(2,RoundingMode.HALF_UP).toPlainString()}.")
                        }
                    } else if (rStMutable.rebalancePosCycleCount >= requiredRebalanceCycles && rStMutable.attemptCount >= MAX_REBALANCE_ATTEMPTS) {
                        mainLoopLogger.warn("⚠️ $symCode: Max standard rebalance attempts ($MAX_REBALANCE_ATTEMPTS) reached for current trigger. Cooldown active or flag will clear if price recovers.")
                    }

                    if (botState.rebalanceState[symCode] != rStMutable) {
                         botState.rebalanceState[symCode] = rStMutable
                         stateChangedThisCycle = true
                    }
                }
            }

            if (stateChangedThisCycle) {
                StateManager.saveState(botState)
            }

            val cycleEndTime = System.currentTimeMillis()
            val elapsedMillis = cycleEndTime - cycleStartTime
            val delayTime = (REFRESH_INTERVAL - elapsedMillis).coerceAtLeast(0L)
            mainLoopLogger.info("----- Cycle End: Took ${elapsedMillis}ms. Active Subs: ${activeSubscriptions.size}. Waiting ${delayTime}ms... -----")
            delay(delayTime)
        }
    } catch (e: CancellationException) {
        mainLoopLogger.info("Main loop cancelled: ${e.message}")
    } catch (e: IllegalStateException) {
        mainLoopLogger.error("Bot Initialization error: ${e.message}", e)
    } catch (e: Exception) {
        mainLoopLogger.error("An unexpected error occurred in main loop: ${e.message}", e)
    } finally {
        mainLoopLogger.info("Cleaning up resources...")
        activeSubscriptions.values.forEach {
            try { it.dispose() } catch (e: Exception) { mainLoopLogger.warn("Error disposing subscription: ${e.message}")}
        }
        activeSubscriptions.clear()
        ExchangeService.cleanup()
        mainLoopLogger.info("Main loop and ExchangeService cleaned up. Exiting.")
    }
}
