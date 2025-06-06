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
import java.time.Duration
import kotlin.math.max
import kotlin.jvm.JvmInline


// --- TrikeShed Core Definitions (Ad-hoc Integration from TrikeShedCore.kt) ---
// ... (TrikeShed Core definitions as in previous step - assuming they are here)
// I. Join and Series Primitives
interface Join<A, B> {
    val a: A
    val b: B
    operator fun component1(): A = a
    operator fun component2(): B = b
    val pair: Pair<A, B> get() = Pair(a, b)
}
private data class _Join<A, B>(override val a: A, override val b: B) : Join<A, B>
infix fun <A, B> A.j(b: B): Join<A, B> = _Join(this, b)
inline val <A, B> Join<A, B>.first: A get() = a
inline val <A, B> Join<A, B>.second: B get() = b
typealias Twin<T> = Join<T, T>
fun <T> T.twin(): Twin<T> = this j this
typealias Series<T> = Join<Int, (Int) -> T>
inline val <T> Series<T>.size: Int get() = a
inline operator fun <T> Series<T>.get(i: Int): T = b(i)
object EmptySeries : Series<Any?> by (0 j { _ -> throw IndexOutOfBoundsException("Accessing element in an empty series.") })
inline fun <T> emptySeries(): Series<T> = EmptySeries as Series<T>

@JvmInline
value class IterableSeries<A>(val s: Series<A>) : Iterable<A>, Series<A> by s {
    override fun iterator(): Iterator<A> = object : Iterator<A> {
        private var index = 0
        override fun hasNext(): Boolean = index < s.size
        override fun next(): A = if (hasNext()) s[index++] else throw NoSuchElementException()
    }
}
inline val <T> Series<T>.`▶`: IterableSeries<T> get() = IterableSeries(this)
fun Series<Char>.asString(): String = this.`▶`.joinToString("")

// III. Tensor Implementation (Core)
typealias Tensor<T> = Join<IntArray, (IntArray) -> T>
inline val <T> Tensor<T>.tensorShape: IntArray get() = a
inline val <T> Tensor<T>.tensorAccessor: (IntArray) -> T get() = b
inline val <T> Tensor<T>.shape: IntArray get() = tensorShape
inline val <T> Tensor<T>.accessor: (IntArray) -> T get() = tensorAccessor
inline val <T> Tensor<T>.tensorRank: Int get() = shape.size
inline val <T> Tensor<T>.rank: Int get() = tensorRank
inline val <T> Tensor<T>.tensorTotalSize: Int get() = if (shape.isEmpty() || shape.any { it == 0 }) 0 else shape.reduce { acc, i -> acc * i }
inline val <T> Tensor<T>.totalSize: Int get() = tensorTotalSize

inline fun <T> TensorConstruct(shape: IntArray, noinline accessor: (IntArray) -> T): Tensor<T> = shape j accessor
inline fun <T> TensorSeries(size: Int, noinline accessor: (Int) -> T): Tensor<T> = intArrayOf(size) j { coords -> accessor(coords[0]) }
inline fun <T> TensorCursor(rows: Int, cols: Int, noinline accessor: (Int, Int) -> T): Tensor<T> = intArrayOf(rows, cols) j { coords -> accessor(coords[0], coords[1]) }

inline operator fun <T> Tensor<T>.invoke(coords: IntArray): T = accessor(coords)
inline operator fun <T> Tensor<T>.invoke(vararg coords: Int): T = accessor(coords)

inline operator fun <T> Tensor<T>.invoke(i: Int): T {
    if (rank == 1) return this(intArrayOf(i))
    if (rank == 0 && totalSize == 1 && i == 0) return this(intArrayOf())
    throw IllegalArgumentException("Single index invoke is for rank 1 Tensors (or scalar Tensor at index 0). Current rank: $rank, totalSize: $totalSize")
}
inline operator fun <T> Tensor<T>.invoke(i: Int, j: Int): T {
    require(rank == 2) { "Two-index invoke is for rank 2 Tensors. Current rank: $rank" }
    return this(intArrayOf(i, j))
}

// IV. Core Tensor Operations
inline infix fun <X, C> Tensor<X>.α(crossinline transform: (X) -> C): Tensor<C> = shape j { coords: IntArray -> transform(accessor(coords)) }
inline infix fun <X, C> Series<X>.α(crossinline transform: (X) -> C): Series<C> = size j { i -> transform(this[i]) }

fun Tensor<*>.linearToCoords(linearIndex: Int): IntArray {
    if (rank == 0) {
        require(linearIndex == 0 && totalSize == 1) { "Linear index for scalar (rank 0) tensor must be 0 and totalSize must be 1." }
        return intArrayOf()
    }
    if (totalSize == 0) throw IllegalArgumentException("Cannot convert linear index for a tensor with totalSize 0 and rank > 0.")
    require(linearIndex >= 0 && linearIndex < totalSize) { "Linear index $linearIndex out of bounds for total size $totalSize (shape ${shape.contentToString()})" }

    val coords = IntArray(rank)
    var remaining = linearIndex
    for (i in rank - 1 downTo 0) {
        val currentDimSize = shape[i]
        coords[i] = remaining % currentDimSize
        remaining /= currentDimSize
    }
    return coords
}

// Column-major
fun Tensor<*>.coordsToLinear(coords: IntArray): Int {
    require(coords.size == rank) { "Coordinate rank mismatch: expected $rank, got ${coords.size} for shape ${shape.contentToString()}" }
    if (rank == 0) return 0
    if (totalSize == 0 && rank > 0) throw IllegalArgumentException("Cannot get linear index for 0-sized tensor with rank > 0")

    var linearIndex = 0
    var multiplier = 1
    for (i in 0 until rank) {
        require(coords[i] >= 0 && coords[i] < shape[i]) { "Coordinate out of bounds: coords[$i]=${coords[i]} for dimension $i with size ${shape[i]}" }
        linearIndex += coords[i] * multiplier
        multiplier *= shape[i]
    }
    return linearIndex
}

fun <T> Tensor<T>.materialize(): Array<T> {
    if (totalSize == 0) {
        @Suppress("UNCHECKED_CAST")
        return arrayOfNulls<Any?>(0) as Array<T>
    }
    @Suppress("UNCHECKED_CAST")
    val arr = arrayOfNulls<Any?>(totalSize) as Array<T>
    for (i in 0 until totalSize) {
        arr[i] = this(linearToCoords(i))
    }
    return arr
}

fun broadcastShapes(shape1: IntArray, shape2: IntArray): IntArray {
    val maxRank = kotlin.math.max(shape1.size, shape2.size)
    val result = IntArray(maxRank)
    for (i in 0 until maxRank) {
        val shape1Idx = shape1.size - 1 - i
        val shape2Idx = shape2.size - 1 - i

        val dim1 = if (shape1Idx >= 0) shape1[shape1Idx] else 1
        val dim2 = if (shape2Idx >= 0) shape2[shape2Idx] else 1

        result[maxRank - 1 - i] = when {
            dim1 == dim2 -> dim1
            dim1 == 1 -> dim2
            dim2 == 1 -> dim1
            else -> throw IllegalArgumentException("Shapes ${shape1.contentToString()} and ${shape2.contentToString()} are not broadcastable at aligned index $i (dims $dim1, $dim2)")
        }
    }
    return result
}

fun <A, B> Tensor<A>.zip(other: Tensor<B>): Tensor<Join<A, B>> {
    val broadcastedShape = broadcastShapes(this.shape, other.shape)
    return TensorConstruct(broadcastedShape) { bCoords ->
        val finalACoords = IntArray(this.rank) { aDimIdx ->
            val bAlignedIdx = bCoords.size - (this.rank - aDimIdx)
            if (aDimIdx >=this.shape.size || this.shape[aDimIdx] == 1) 0 else bCoords[bAlignedIdx] // Check aDimIdx bounds
        }
        val finalBCoords = IntArray(other.rank) { bDimIdx ->
            val bAlignedIdx = bCoords.size - (other.rank - bDimIdx)
            if (bDimIdx >= other.shape.size || other.shape[bDimIdx] == 1) 0 else bCoords[bAlignedIdx] // Check bDimIdx bounds
        }
        this(finalACoords) j other(finalBCoords)
    }
}

inline fun <A, B, C> Tensor<A>.combine(other: Tensor<B>, crossinline transform: (A, B) -> C): Tensor<C> {
    val broadcastedShape = broadcastShapes(this.shape, other.shape)
    return TensorConstruct(broadcastedShape) { bCoords ->
        val finalACoords = IntArray(this.rank) { aDimIdx -> val bAlignedIdx = bCoords.size - (this.rank - aDimIdx) ; if (aDimIdx >=this.shape.size || this.shape[aDimIdx] == 1) 0 else bCoords[bAlignedIdx] }
        val finalBCoords = IntArray(other.rank) { bDimIdx -> val bAlignedIdx = bCoords.size - (other.rank - bDimIdx) ; if (bDimIdx >=other.shape.size ||other.shape[bDimIdx] == 1) 0 else bCoords[bAlignedIdx] }
        transform(this(finalACoords), other(finalBCoords))
    }
}

// V. CoreTensorCursor Layer (Typealiases and Basic Accessors)
typealias CoreTensorCursor<T> = Tensor<T>
typealias CoreTensorRowVec<T> = Tensor<T>
typealias CoreTensorColumnVec<T> = Tensor<T>

// VI. Metadata Types (from TrikeShedCore)
interface TypeMemento { val networkSize: Int? }
enum class IOMemento : TypeMemento {
    IoByte, IoShort, IoInt, IoFloat, IoDouble, IoLong,
    IoBoolean, IoChar, IoString, IoCharSeries, IoBigDecimal,
    IoBigInt, IoDateTime, IoDuration, IoUUID, IoBinary, IoUnknown;
    override val networkSize: Int? get() = null
}
typealias ColumnMeta = Join<String, TypeMemento>
inline val ColumnMeta.name: String get() = a
inline val ColumnMeta.type: TypeMemento get() = b
typealias CursorMeta = Tensor<ColumnMeta>
typealias CoreTensorCursorWithMeta<T> = Join<CoreTensorCursor<T>, CursorMeta>

inline val <T> CoreTensorCursor<T>.rows: Int get() = if (rank >= 1) shape[0] else if (rank == 0 && totalSize == 1) 1 else 0
inline val <T> CoreTensorCursor<T>.cols: Int get() = if (rank >= 2) shape[1] else if (rank == 1 && shape[0] == 0) 0 else if (rank == 1) 1 else if (rank == 0 && totalSize == 1) 1 else 0


fun <T> CoreTensorCursor<T>.row(index: Int): CoreTensorRowVec<T> {
    require(rank == 2) { "Cursor must be rank 2 for row access. Rank is $rank, shape ${shape.contentToString()}" }
    require(index in 0 until rows) { "Row index $index out of bounds for rows $rows" }
    return TensorSeries(cols) { colIdx -> this(index, colIdx) }
}
fun <T> CoreTensorCursor<T>.col(index: Int): CoreTensorColumnVec<T> {
    require(rank == 2) { "Cursor must be rank 2 for column access. Rank is $rank, shape ${shape.contentToString()}" }
    require(index in 0 until cols) { "Column index $index out of bounds for cols $cols" }
    return TensorSeries(rows) { rowIdx -> this(rowIdx, index) }
}

operator fun <T> CoreTensorCursor<T>.get(rowRange: IntRange): CoreTensorCursor<T> {
    require(rank == 2) { "Cursor must be rank 2 for row range slicing. Rank: $rank, Shape: ${shape.contentToString()}" }
    if (rows == 0 && (rowRange.first == 0 && rowRange.last == -1 || rowRange.isEmpty())) return TensorCursor(0, cols) { _, _ -> throw IndexOutOfBoundsException("Empty range on empty cursor")}
    if (rows == 0) return TensorCursor(0, cols) { _, _ -> throw IndexOutOfBoundsException("Cannot slice rows of an empty cursor (0 rows)") }
    val safeFirst = rowRange.first.coerceIn(0, rows -1)
    val safeLast = rowRange.last.coerceIn(0, rows -1)
    val newRows = if (safeFirst > safeLast) 0 else safeLast - safeFirst + 1
    if (newRows == 0) return TensorCursor(0, cols) { _, _ -> throw IndexOutOfBoundsException("Empty range after coercion results in zero rows to slice.")}
    return TensorCursor(newRows, cols) { r, c -> this(safeFirst + r, c) }
}

operator fun <T> CoreTensorCursor<T>.get(vararg colIndices: Int): CoreTensorCursor<T> {
    require(rank == 2) { "Cursor must be rank 2 for column indexing. Rank: $rank, Shape: ${shape.contentToString()}" }
    if (cols == 0) {
        require(colIndices.isEmpty()) { "Cannot select columns from a cursor with 0 columns unless selecting 0 columns."}
        return TensorCursor(rows, 0) {_,_ -> throw IndexOutOfBoundsException("Selected 0 columns from 0-column cursor.")}
    }
    colIndices.forEach { require(it >= 0 && it < cols) { "Column index $it out of bounds for cols $cols" } }
    val newCols = colIndices.size
    return TensorCursor(rows, newCols) { r, c -> this(r, colIndices[c]) }
}

inline val <T> CoreTensorCursorWithMeta<T>.coreTensorMeta: CursorMeta get() = b
inline val <T> CoreTensorCursorWithMeta<T>.meta: CursorMeta get() = b
inline val CursorMeta.names: List<String> get() = if (this.totalSize == 0 || this.rank == 0) emptyList() else this.`▶`.map { it.name }

// --- Helper Series Operations ---
fun Series<BigDecimal?>.sumOrNull(): BigDecimal? {
    if (this.size == 0) return null
    var sum = BigDecimal.ZERO
    var hasNonNull = false
    this.`▶`.forEach { value ->
        if (value != null) {
            sum = sum.add(value)
            hasNonNull = true
        }
    }
    return if (hasNonNull) sum else null
}

fun <A, B> Series<A>.zip(other: Series<B>): Series<Pair<A, B>> {
    require(this.size == other.size) { "Series must have the same size to zip. Sizes: ${this.size} and ${other.size}" }
    return (this.size j { i -> Pair(this[i], other[i]) })
}

// --- End TrikeShed Core Definitions ---

// --- Bot Specific Typealiases ---
typealias PriceSeries = Series<java.math.BigDecimal?>
typealias QuantitySeries = Series<java.math.BigDecimal>
typealias ValueSeries = Series<java.math.BigDecimal?>
typealias DeviationSeries = Series<Double?>
typealias BaselineSeries = Series<Double?>
typealias SymbolSeriesS = Series<String>
typealias CurrencySeries = Series<org.knowm.xchange.currency.Currency>
typealias EligibilitySeries = Series<Boolean>
typealias IndexSeries = Series<Int>

typealias StringDoubleMap = MutableMap<String, Double>
typealias StringTrailingDataMap = MutableMap<String, TrailingData>
typealias StringRebalanceDataMap = MutableMap<String, RebalanceData>
typealias StringADZStateMap = MutableMap<String, Boolean>
typealias StringLongMap = MutableMap<String, Long>

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
    val baselines: StringDoubleMap = mutableMapOf(),
    val trailingState: StringTrailingDataMap = mutableMapOf(),
    val lastActionTimestamps: StringLongMap = mutableMapOf(),
    val rebalanceState: StringRebalanceDataMap = mutableMapOf(),
    val adaptiveDeadZoneState: StringADZStateMap = mutableMapOf()
)

// --- Global State Variable ---
lateinit var botState: BotState

// --- Portfolio Tensor Definitions ---
val portfolioColumnNames = listOf("Symbol", "CurrencyObj", "Quantity", "Price", "Value", "Baseline", "Deviation", "AbsDifference", "PriceChange")

val portfolioMetaDefinition: List<ColumnMeta> = listOf(
    "Symbol" j IOMemento.IoString,
    "CurrencyObj" j IOMemento.IoUnknown,
    "Quantity" j IOMemento.IoBigDecimal,
    "Price" j IOMemento.IoBigDecimal,
    "Value" j IOMemento.IoBigDecimal,
    "Baseline" j IOMemento.IoDouble,
    "Deviation" j IOMemento.IoDouble,
    "AbsDifference" j IOMemento.IoBigDecimal,
    "PriceChange" j IOMemento.IoBigDecimal
)

val portfolioCursorMeta: CursorMeta = TensorSeries(portfolioMetaDefinition.size) { i -> portfolioMetaDefinition[i] }
typealias PortfolioTensor = CoreTensorCursorWithMeta<Any?>


// --- Strategy Constants and Top-Level Variables ---
const val LIVE_TRADING_ENABLED = false // MASTER SWITCH FOR SIMULATION VS LIVE
val QUOTE_CURRENCY_CODE = "USD"
val QUOTE_CURRENCY = Currency(QUOTE_CURRENCY_CODE)

val HARVEST_EXCLUDE_SYMBOLS = setOf("BTC", "USDC", QUOTE_CURRENCY_CODE)
const val FLAT_HARVEST_TRIGGER_PERCENT = 0.03
const val HARVEST_CYCLE_THRESHOLD = 3
const val MIN_SURPLUS_FOR_HARVEST = 1.00
const val MIN_SURPLUS_FOR_FORCED_HARVEST = 1.00
const val FORCED_HARVEST_TIMEOUT = 20 * 60 * 1000L
const val TARGET_ADJUST_PERCENT = 0.000

const val ENABLE_PORTFOLIO_HARVEST = true
const val PORTFOLIO_HARVEST_TRIGGER_DEVIATION_PERCENT = 0.05
const val PORTFOLIO_HARVEST_CONFIRMATION_CYCLES = 3
const val MIN_ASSET_SURPLUS_FOR_PORTFOLIO_HARVEST = 0.10
val REBALANCE_EXCLUDE_SYMBOLS = setOf("USDC", QUOTE_CURRENCY_CODE)

const val HARVEST_ALLOC_REINVEST_PERCENT = 0.50
const val HARVEST_ALLOC_CASH_PERCENT = 0.40
const val HARVEST_ALLOC_BTC_PERCENT = 0.10
const val MIN_HARVEST_TO_ALLOCATE = 1.00
const val MIN_NEGATIVE_DEVIATION_FOR_REINVEST = -0.01
const val MIN_REINVEST_BUY_USD = 0.50
const val MIN_BTC_BUY_USD = 1.00

const val FLAT_REBALANCE_TRIGGER_PERCENT = 0.04
const val PARTIAL_RECOVERY_PERCENT = 0.875
const val REBALANCE_POSITIVE_THRESHOLD = 3
const val MAX_REBALANCE_ATTEMPTS = 3
const val REBALANCE_COOLDOWN = 30 * 60 * 1000L
const val FORCE_REBALANCE_TIMEOUT = 25 * 60 * 1000L
const val FORCE_REBALANCE_SHORTFALL_PERCENT = 0.25
const val MIN_PARTIAL_REBALANCE_USD = 1.00
const val MIN_FORCED_REBALANCE_USD = 1.00

const val ENABLE_ADAPTIVE_DEAD_ZONE = true
const val ADAPTIVE_DZ_INACTIVITY_TIMEOUT = 3 * 60 * 60 * 1000L
const val ADAPTIVE_DZ_HARVEST_TRIGGER_PERCENT = 0.020
const val ADAPTIVE_DZ_REBALANCE_TRIGGER_PERCENT = 0.020

const val ENABLE_CRASH_PROTECTION = true
const val CP_TRIGGER_ASSET_PERCENT = 0.70
const val CP_TRIGGER_MIN_NEGATIVE_DEV_PERCENT = -0.01
const val CRASH_PROTECTION_THRESHOLD_INCREASE = 2
const val CRASH_PROTECTION_PARTIAL_RECOVERY_PERCENT_FACTOR = 0.55 / 0.875


val latestPrices = mutableMapOf<CurrencyPair, Ticker>()
data class PriceTick(val price: BigDecimal, val timestamp: Long)
val latestPriceInfo = mutableMapOf<CurrencyPair, Pair<PriceTick?, PriceTick?>>()

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
    val absoluteDifference: BigDecimal?,
    val priceChange: BigDecimal? = null
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
            val json = Json { prettyPrint = true; encodeDefaults = true; ignoreUnknownKeys = true }
            val jsonString = json.encodeToString(botState)
            File(StateManager.STATE_FILE_PATH).writeText(jsonString)
            shutdownLogger.info("Bot state saved to ${StateManager.STATE_FILE_PATH} during shutdown.")
        } catch (e: Exception) {
            shutdownLogger.error("CRITICAL ERROR: Failed to save state during shutdown: ${e.message}", e)
        }
    } else {
        shutdownLogger.info("botState not initialized, no state to save during shutdown.")
    }

    if (ExchangeService.isExchangeInitialized()) {
         ExchangeService.cleanup()
         shutdownLogger.info("ExchangeService cleanup called during shutdown.")
    }
    shutdownLogger.info("Shutdown hook finished.")
}

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

// --- OrderSimulator Object ---
object OrderSimulator {
    private val logger = LoggerFactory.getLogger(OrderSimulator::class.java)

    private val simulatedHoldings: MutableMap<Currency, BigDecimal> = mutableMapOf()
    private var simulatedCash: BigDecimal = BigDecimal.ZERO
    var isInitialized: Boolean = false
        private set

    fun initializeBalances(
        initialCash: BigDecimal,
        initialHoldings: Map<Currency, BigDecimal>
    ) {
        simulatedCash = initialCash
        simulatedHoldings.clear()
        simulatedHoldings.putAll(initialHoldings)
        isInitialized = true
        logger.info("OrderSimulator initialized. Cash: $simulatedCash ${QUOTE_CURRENCY_CODE}, Holdings: $simulatedHoldings")
    }

    fun getSimulatedBalance(currency: Currency): BigDecimal {
        return if (currency == QUOTE_CURRENCY) {
            simulatedCash
        } else {
            simulatedHoldings.getOrDefault(currency, BigDecimal.ZERO)
        }
    }

    fun getAllSimulatedHoldings(): Map<Currency, BigDecimal> {
        return simulatedHoldings.toMap()
    }

    fun executeSimulatedMarketBuy(
        pair: CurrencyPair,
        orderAmount: OrderAmount,
        currentPrice: BigDecimal
    ): String? {
        if (!isInitialized) {
            logger.error("SIMULATOR ERROR: OrderSimulator not initialized with balances.")
            return null
        }
        if (currentPrice <= BigDecimal.ZERO) {
            logger.error("SIMULATOR ERROR: Invalid current price $currentPrice for BUY ${pair.base}/${pair.counter}")
            return null
        }

        val baseCurrency = pair.base
        val quoteCurrency = pair.counter

        val baseQuantityToBuy: BigDecimal
        val costInCash: BigDecimal

        when (orderAmount) {
            is OrderAmount.QuoteSize -> {
                val quoteSize = orderAmount.amount
                if (quoteSize <= BigDecimal.ZERO) {
                    logger.warn("SIMULATOR: BUY order for $pair has zero or negative quote size: $quoteSize. Skipping.")
                    return null
                }
                val baseCurrencyScale = assetPairMetaData[pair]?.baseScale
                                    ?: assetExchangeMetaData[pair.base]?.scale
                                    ?: 8
                baseQuantityToBuy = quoteSize.divide(currentPrice, baseCurrencyScale, RoundingMode.DOWN)
                costInCash = quoteSize
            }
            is OrderAmount.BaseSize -> {
                baseQuantityToBuy = orderAmount.amount
                if (baseQuantityToBuy <= BigDecimal.ZERO) {
                    logger.warn("SIMULATOR: BUY order for $pair has zero or negative base size: $baseQuantityToBuy. Skipping.")
                    return null
                }
                costInCash = baseQuantityToBuy.multiply(currentPrice)
            }
        }

        if (baseQuantityToBuy <= BigDecimal.ZERO) {
             logger.warn("SIMULATOR: Calculated base quantity for BUY $baseQuantityToBuy is zero or negative. Skipping.")
             return null
        }

        if (simulatedCash < costInCash) {
            logger.warn("SIMULATOR: Insufficient simulated cash for BUY $baseQuantityToBuy ${baseCurrency}. Need $costInCash $quoteCurrency, have $simulatedCash $quoteCurrency.")
            return null
        }

        simulatedCash = simulatedCash.subtract(costInCash)
        simulatedHoldings[baseCurrency] = simulatedHoldings.getOrDefault(baseCurrency, BigDecimal.ZERO).add(baseQuantityToBuy)

        val simulatedOrderId = "sim_buy_${System.nanoTime()}"
        logger.info("SIMULATED TRADE: BUY $baseQuantityToBuy $baseCurrency @ $currentPrice $quoteCurrency (Cost: ~$costInCash $quoteCurrency). Order ID: $simulatedOrderId")
        logger.info("SIMULATOR BALANCES after BUY: Cash: $simulatedCash $quoteCurrency, ${baseCurrency}: ${simulatedHoldings[baseCurrency]}")
        return simulatedOrderId
    }

    fun executeSimulatedMarketSell(
        pair: CurrencyPair,
        baseQuantityToSell: BigDecimal,
        currentPrice: BigDecimal
    ): String? {
        if (!isInitialized) {
            logger.error("SIMULATOR ERROR: OrderSimulator not initialized with balances.")
            return null
        }
        if (currentPrice <= BigDecimal.ZERO) {
            logger.error("SIMULATOR ERROR: Invalid current price $currentPrice for SELL ${pair.base}/${pair.counter}")
            return null
        }
        if (baseQuantityToSell <= BigDecimal.ZERO) {
            logger.warn("SIMULATOR: SELL order for ${pair.base} has zero or negative base size: $baseQuantityToSell. Skipping.")
            return null
        }

        val baseCurrency = pair.base
        val quoteCurrency = pair.counter

        val currentBaseHolding = simulatedHoldings.getOrDefault(baseCurrency, BigDecimal.ZERO)
        if (currentBaseHolding < baseQuantityToSell) {
            logger.warn("SIMULATOR: Insufficient simulated $baseCurrency for SELL. Need $baseQuantityToSell, have $currentBaseHolding.")
            return null
        }

        val proceedsInCash = baseQuantityToSell.multiply(currentPrice)
        simulatedHoldings[baseCurrency] = currentBaseHolding.subtract(baseQuantityToSell)
        simulatedCash = simulatedCash.add(proceedsInCash)

        val simulatedOrderId = "sim_sell_${System.nanoTime()}"
        logger.info("SIMULATED TRADE: SELL $baseQuantityToSell $baseCurrency @ $currentPrice $quoteCurrency (Proceeds: ~$proceedsInCash $quoteCurrency). Order ID: $simulatedOrderId")
        logger.info("SIMULATOR BALANCES after SELL: Cash: $simulatedCash $quoteCurrency, ${baseCurrency}: ${simulatedHoldings[baseCurrency]}")

        if ((simulatedHoldings[baseCurrency]?.compareTo(BigDecimal.valueOf(1e-18)) ?: 0) < 0) {
            simulatedHoldings.remove(baseCurrency)
            logger.info("SIMULATOR: Removed $baseCurrency from simulated holdings as quantity is effectively zero.")
        }
        return simulatedOrderId
    }
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
    private lateinit var exchangeVar: Exchange
    private val disposables = CompositeDisposable()

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
        exchangeVar = ExchangeFactory.INSTANCE.createExchange(exchangeSpecification)
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
        // If OrderSimulator is active and initialized, return its balances
        if (!LIVE_TRADING_ENABLED && OrderSimulator.isInitialized) { // Check LIVE_TRADING_ENABLED flag
            val simBalances = mutableMapOf<Currency, Balance>()
            simBalances[QUOTE_CURRENCY] = Balance.Builder().currency(QUOTE_CURRENCY).available(OrderSimulator.getSimulatedBalance(QUOTE_CURRENCY)).total(OrderSimulator.getSimulatedBalance(QUOTE_CURRENCY)).build()
            OrderSimulator.getAllSimulatedHoldings().forEach { (currency, amount) -> // Use new getter
                 simBalances[currency] = Balance.Builder().currency(currency).available(amount).total(amount).build()
            }
            logger.debug("Returning SIMULATED account balances: $simBalances")
            return simBalances
        }
        // Otherwise, fetch from exchange
        return withContext(Dispatchers.IO) {
            try {
                val accountInfo = accountService.accountInfo
                logger.debug("Fetched REAL account info: $accountInfo")
                accountInfo?.getWallet()?.balances?.values?.associateBy { it.currency }
                    ?: accountInfo?.wallets?.values?.flatMap { it.balances.values }?.associateBy { it.currency }
            } catch (e: Exception) {
                logger.error("Error fetching REAL account balances: ${e.message}", e)
                null
            }
        }
    }

    suspend fun getProductDetails(pair: CurrencyPair): CurrencyPairMetaData? {
        return withContext(Dispatchers.IO) {
            try {
                val metaData = exchangeMetaData ?: exchangeVar.remoteInit()
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
        if (!LIVE_TRADING_ENABLED) {
            val currentPrice = latestPrices[pair]?.last
            if (currentPrice == null || currentPrice <= BigDecimal.ZERO) {
                logger.error("SIMULATOR: Cannot place order for $pair, current price is unknown or invalid: $currentPrice")
                return null
            }
            logger.info("SIMULATOR: placeMarketOrder directed to OrderSimulator.")
            return if (type == Order.OrderType.BID) {
                OrderSimulator.executeSimulatedMarketBuy(pair, orderAmount, currentPrice)
            } else {
                when (orderAmount) {
                    is OrderAmount.BaseSize -> OrderSimulator.executeSimulatedMarketSell(pair, orderAmount.amount, currentPrice)
                    is OrderAmount.QuoteSize -> {
                        logger.error("SIMULATOR: SELL order with QuoteSize is not directly supported by simulator's sell method, requires base size.")
                        null
                    }
                }
            }
        } else { // Actual Live Trading
            logger.warn("LIVE TRADING: Attempting to place real market order.")
            // TODO: Implement actual XChange tradeService.placeMarketOrder call here
            // This requires converting OrderAmount to the expected BigDecimal originalAmount (base currency)
            // and ensuring all parameters are correct for the live API.
            // For now, it will still simulate to avoid accidental live trades.
            logger.warn("LIVE TRADING: Market order placement NOT YET FULLY IMPLEMENTED. Simulating for safety.")
            val currentPrice = latestPrices[pair]?.last ?: BigDecimal.ONE // Fallback price for simulation
            return if (type == Order.OrderType.BID) {
                OrderSimulator.executeSimulatedMarketBuy(pair, orderAmount, currentPrice)
            } else {
                when (orderAmount) {
                    is OrderAmount.BaseSize -> OrderSimulator.executeSimulatedMarketSell(pair, orderAmount.amount, currentPrice)
                    else -> { logger.error("Live SELL with QuoteSize needs conversion first."); null }
                }
            }
            // return "live_order_id_placeholder"; // Replace with actual tradeService call
        }
    }

    fun subscribeToPriceTicks(pair: CurrencyPair, onPriceUpdate: (Ticker) -> Unit): Disposable? {
        if (!this::exchangeVar.isInitialized || exchangeVar !is StreamingExchange) {
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
        if (this::exchangeVar.isInitialized && exchangeVar is StreamingExchange) {
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


// --- Main Function ---
fun main() = runBlocking {
    Runtime.getRuntime().addShutdownHook(shutdownHook)

    mainLoopLogger.info("Coinbase XChange Bot Starting...")

    System.setProperty("org.slf4j.simpleLogger.defaultLogLevel", "info")
    System.setProperty("org.slf4j.simpleLogger.log.StateManager", "info")
    System.setProperty("org.slf4j.simpleLogger.log.ExchangeService", "info")
    System.setProperty("org.slf4j.simpleLogger.log.MainLoopLogic", "info")
    System.setProperty("org.slf4j.simpleLogger.log.OrderSimulator", "info")
    System.setProperty("org.slf4j.simpleLogger.log.org.knowm.xchange", "warn")
    System.setProperty("org.slf4j.simpleLogger.log.org.knowm.xchange.coinbasepro", "info")
    System.setProperty("org.slf4j.simpleLogger.showDateTime", "true")
    System.setProperty("org.slf4j.simpleLogger.dateTimeFormat", "yyyy-MM-dd HH:mm:ss:SSS Z")
    System.setProperty("org.slf4j.simpleLogger.showThreadName", "true")

    botState = StateManager.loadState()
    mainLoopLogger.info("Initial bot state loaded: ${botState.baselines.size} baselines, ${botState.trailingState.size} trailing states, ${botState.lastActionTimestamps.size} timestamps, ${botState.rebalanceState.size} rebalance states, ${botState.adaptiveDeadZoneState.size} ADZ states.")

    var activeSubscriptions = mutableMapOf<CurrencyPair, Disposable>()
    val REFRESH_INTERVAL = 8000L
    var currentPortfolioDeviationPercentForDisplay = 0.0
    var validPortfolioItemsForTrading = listOf<PortfolioRow>()
    var cycleCount = 0

    try {
        ExchangeService.initialize() // Initializes real exchange connection for metadata, market data
        mainLoopLogger.info("ExchangeService initialized.")

        if (!LIVE_TRADING_ENABLED) {
            if (!OrderSimulator.isInitialized) {
                mainLoopLogger.info("SIMULATED MODE: Initializing OrderSimulator with live balances as seed...")
                val initialRealBalances = ExchangeService.getAccountBalances() // Fetch once
                if (initialRealBalances != null) {
                    val initialRealCash = initialRealBalances[QUOTE_CURRENCY]?.available ?: BigDecimal("10000.00") // Default if no quote currency
                    val initialRealHoldings = initialRealBalances
                        .filterKeys { it != QUOTE_CURRENCY }
                        .mapValues { it.value.total ?: BigDecimal.ZERO } // Use total for holdings
                    OrderSimulator.initializeBalances(initialRealCash, initialRealHoldings)
                    mainLoopLogger.info("OrderSimulator seeded with balances from exchange. Cash: $initialRealCash, Holdings: $initialRealHoldings")
                } else {
                    mainLoopLogger.warn("Failed to fetch initial real balances to seed simulator. Initializing simulator with default fallback values.")
                    OrderSimulator.initializeBalances(BigDecimal("10000.00"), emptyMap())
                }
            } else {
                mainLoopLogger.info("SIMULATED MODE: OrderSimulator already initialized. Continuing with existing simulated balances.")
            }
        }


        while (true) {
            cycleCount++
            if (cycleCount > 3) {
                mainLoopLogger.info("TEST EXECUTION: Reached ${cycleCount - 1} cycles, exiting.")
                break
            }
            mainLoopLogger.info("TEST EXECUTION: Starting cycle $cycleCount...")

            val cycleStartTime = System.currentTimeMillis()
            mainLoopLogger.info("----- Cycle Start: ${Instant.ofEpochMilli(cycleStartTime)} (Cycle #$cycleCount) -----")
            harvestedAmountThisCycle = BigDecimal.ZERO
            anyTradesThisCycle = false
            var stateChangedThisCycle = false

            var loopCashBalance: BigDecimal
            val loopCurrentHoldings: MutableMap<Currency, BigDecimal>

            if (LIVE_TRADING_ENABLED) {
                mainLoopLogger.info("LIVE TRADING MODE: Fetching live balances...")
                val liveBalances = ExchangeService.getAccountBalances() // Still calls simulator if !LIVE_TRADING_ENABLED
                if (liveBalances == null) {
                    mainLoopLogger.error("Failed to fetch LIVE account balances. Skipping cycle.")
                    delay(REFRESH_INTERVAL)
                    continue
                }
                loopCashBalance = BigDecimal.ZERO
                loopCurrentHoldings = mutableMapOf()
                liveBalances.forEach { (currency, balance) ->
                    if (currency == QUOTE_CURRENCY) {
                        loopCashBalance = loopCashBalance.add(balance.available ?: BigDecimal.ZERO)
                    } else {
                        val totalQty = balance.total ?: BigDecimal.ZERO
                        if (totalQty > BigDecimal.ZERO) {
                            loopCurrentHoldings[currency] = totalQty
                        }
                    }
                }
            } else { // SIMULATED MODE
                mainLoopLogger.info("SIMULATED TRADING MODE: Using simulated balances...")
                loopCashBalance = OrderSimulator.getSimulatedBalance(QUOTE_CURRENCY)
                loopCurrentHoldings = mutableMapOf()
                OrderSimulator.getAllSimulatedHoldings().forEach { (curr, qty) ->
                    if (qty > BigDecimal.ZERO) {
                        loopCurrentHoldings[curr] = qty
                    }
                }
            }

            mainLoopLogger.info("Effective Cash Balance for cycle: $QUOTE_CURRENCY_CODE ${loopCashBalance.toPlainString()}")
            if (loopCurrentHoldings.isNotEmpty()) {
                mainLoopLogger.info("Effective Holdings for cycle: ${loopCurrentHoldings.entries.joinToString { it.key.currencyCode + ": " + it.value.toPlainString() }}")
            } else {
                mainLoopLogger.info("No significant crypto holdings for cycle.")
            }

            val symbolsToTrack = loopCurrentHoldings.keys.toMutableSet() // Use loopCurrentHoldings
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
                latestPriceInfo.remove(pair)
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
                    val newPriceTick = PriceTick(ticker.last, ticker.timestamp?.time ?: System.currentTimeMillis())
                    synchronized(latestPriceInfo) {
                        val existingHistory = latestPriceInfo[ticker.currencyPair]
                        latestPriceInfo[ticker.currencyPair] = Pair(newPriceTick, existingHistory?.first)
                    }
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

            // --- Transform loopCurrentHoldings into Series ---
            val heldCurrencyList = loopCurrentHoldings.keys.toList()
            val heldQuantityList = loopCurrentHoldings.values.toList()

            val numHeldAssets = heldCurrencyList.size
            val heldCurrenciesSeries: CurrencySeries = TensorSeries(numHeldAssets) { i -> heldCurrencyList[i] }
            val heldQuantitiesSeries: QuantitySeries = TensorSeries(numHeldAssets) { i -> heldQuantityList[i] }

            // ... (The rest of the portfolio calculation, display, ADZ/CP, and strategy logic using these series and loopCashBalance) ...
            // This means replacing `currentHoldings` with `loopCurrentHoldings` and `cashBalance` with `loopCashBalance`
            // in all subsequent calculations and logging within the current cycle.
            // The strategy logic itself (Portfolio Harvest, Individual Harvest, Allocation, Rebalance)
            // will use `validPortfolioItemsForTrading` which is derived from these series,
            // and will use `loopCashBalance` for checking available funds.

            // --- Calculate Derived Series (using loop's data) ---
            val currentPricesList = mutableListOf<BigDecimal?>()
            val previousPricesList = mutableListOf<BigDecimal?>()
            val baselinesList = mutableListOf<Double?>()
            for (i in 0 until numHeldAssets) {
                val currency = heldCurrenciesSeries[i]; val pair = CurrencyPair(currency, QUOTE_CURRENCY)
                currentPricesList.add(latestPriceInfo[pair]?.first?.price)
                previousPricesList.add(latestPriceInfo[pair]?.second?.price)
                baselinesList.add(botState.baselines[currency.currencyCode])
            }
            val currentPricesSeries: PriceSeries = TensorSeries(numHeldAssets) { i -> currentPricesList[i] }
            val previousPricesSeries: PriceSeries = TensorSeries(numHeldAssets) { i -> previousPricesList[i] }
            val baselinesSeries: BaselineSeries = TensorSeries(numHeldAssets) { i -> baselinesList[i] }

            val currentValuesSeries: ValueSeries = heldQuantitiesSeries.zip(currentPricesSeries).α { (qty, price) -> if (price != null && price > BigDecimal.ZERO) qty.multiply(price) else null }
            val absoluteDifferencesSeries: Series<BigDecimal?> = currentValuesSeries.zip(baselinesSeries).α { (v, b) -> if (v != null && b != null) v.subtract(BigDecimal.valueOf(b)) else null }
            val deviationsSeries: DeviationSeries = absoluteDifferencesSeries.zip(baselinesSeries).α { (ad, b) -> if (ad != null && b != null && b > 0.0) ad.divide(BigDecimal.valueOf(b), MathContext.DECIMAL64).toDouble() else null }
            val priceChangesSeries: PriceSeries = currentPricesSeries.zip(previousPricesSeries).α { (curr, prev) -> if (curr != null && prev != null) curr.subtract(prev) else null }
            totalHoldingsValue = currentValuesSeries.sumOrNull() ?: BigDecimal.ZERO

            // Baseline Init/Verify (uses loop's data, updates global botState)
            var baselinesVerifiedOrSetThisCycleForInit = false
             for (i in 0 until numHeldAssets) {
                val symbolCode = heldCurrenciesSeries[i].currencyCode
                val currentHoldingValueBD = currentValuesSeries[i]
                var baselineValue = botState.baselines[symbolCode]
                if (currentHoldingValueBD != null && currentHoldingValueBD > BigDecimal.ZERO) {
                    if (!initialized) {
                        if (baselineValue != null && baselineValue > 0.01) { baselinesVerifiedOrSetThisCycleForInit = true }
                        else if (baselineValue == null && currentHoldingValueBD > BigDecimal.valueOf(0.01)) {
                            botState.baselines[symbolCode] = currentHoldingValueBD.toDouble()
                            mainLoopLogger.info("✨ Initialized baseline $symbolCode: $${currentHoldingValueBD.toDouble()} (First cycle).")
                            baselinesVerifiedOrSetThisCycleForInit = true; stateChangedThisCycle = true
                        }
                    }
                    if (initialized && baselineValue == null && currentHoldingValueBD > BigDecimal.valueOf(0.01)) {
                        botState.baselines[symbolCode] = currentHoldingValueBD.toDouble()
                        mainLoopLogger.info("✨ Initialized baseline $symbolCode (post-init): $${currentHoldingValueBD.toDouble()}.")
                        stateChangedThisCycle = true
                    }
                    val currentBaselineForTimestamp = botState.baselines[symbolCode]
                    if (botState.lastActionTimestamps[symbolCode] == null && currentBaselineForTimestamp != null && currentBaselineForTimestamp > 0.01) {
                        botState.lastActionTimestamps[symbolCode] = System.currentTimeMillis()
                        mainLoopLogger.info("✨ Initialized last action timestamp for $symbolCode.")
                        stateChangedThisCycle = true
                    }
                }
            }
            if (!initialized) {
                if (baselinesVerifiedOrSetThisCycleForInit) { mainLoopLogger.info("✅ Baselines & Timestamps init/verify complete."); initialized = true }
                else if (loopCurrentHoldings.isNotEmpty() && (0 until numHeldAssets).all { currentPricesSeries[it] == null } && !baselinesVerifiedOrSetThisCycleForInit) { mainLoopLogger.info("⏳ Waiting for prices for baseline init (all holdings lack prices)...") }
                else if (loopCurrentHoldings.isEmpty()) { mainLoopLogger.info("✅ No holdings, baseline init considered complete."); initialized = true }
            }

            val portfolioDataTensorPart: CoreTensorCursor<Any?> = TensorCursor(numHeldAssets, portfolioColumnNames.size) { r,c -> /* ... as before ... */
                 when (portfolioColumnNames[c]) {
                    "Symbol" -> heldCurrenciesSeries[r].currencyCode
                    "CurrencyObj" -> heldCurrenciesSeries[r]
                    "Quantity" -> heldQuantitiesSeries[r]
                    "Price" -> currentPricesSeries[r]
                    "Value" -> currentValuesSeries[r]
                    "Baseline" -> botState.baselines[heldCurrenciesSeries[r].currencyCode]
                    "Deviation" -> deviationsSeries[r]
                    "AbsDifference" -> absoluteDifferencesSeries[r]
                    "PriceChange" -> priceChangesSeries[r]
                    else -> throw IndexOutOfBoundsException("Invalid column name at index $c for portfolio tensor")
                }
            }
            val currentPortfolioTensor: PortfolioTensor = portfolioDataTensorPart j portfolioCursorMeta

            val currentSymbolsInPortfolioView = (0 until heldCurrenciesSeries.size).map { heldCurrenciesSeries[it].currencyCode }.toSet()
            val symbolsToRemove = botState.baselines.keys.filterNot { it in currentSymbolsInPortfolioView }.toSet()
            if (symbolsToRemove.isNotEmpty()) { /* ... state cleanup as before ... */
                symbolsToRemove.forEach { symCode ->
                    mainLoopLogger.info("🗑️ Clearing state for sold/removed asset: $symCode.")
                    botState.baselines.remove(symCode); botState.trailingState.remove(symCode); botState.lastActionTimestamps.remove(symCode); botState.rebalanceState.remove(symCode); botState.adaptiveDeadZoneState.remove(symCode)
                }
                stateChangedThisCycle = true
            }

            val tempPortfolioRows = mutableListOf<PortfolioRow>()
            if (currentPortfolioTensor.a.rows > 0) { /* ... rebuild tempPortfolioRows as before ... */
                 for (r in 0 until currentPortfolioTensor.a.rows) {
                    val symbol = currentPortfolioTensor.a(r, portfolioColumnNames.indexOf("Symbol")) as String
                    val currency = currentPortfolioTensor.a(r, portfolioColumnNames.indexOf("CurrencyObj")) as Currency
                    val quantity = currentPortfolioTensor.a(r, portfolioColumnNames.indexOf("Quantity")) as BigDecimal
                    val price = currentPortfolioTensor.a(r, portfolioColumnNames.indexOf("Price")) as? BigDecimal
                    val actualBaseline = botState.baselines[symbol]
                    val value = currentPortfolioTensor.a(r, portfolioColumnNames.indexOf("Value")) as? BigDecimal
                    val priceChange = currentPortfolioTensor.a(r, portfolioColumnNames.indexOf("PriceChange")) as? BigDecimal
                    val actualDeviation = if (value != null && actualBaseline != null && actualBaseline > 0.0 && price != null && price > BigDecimal.ZERO) value.subtract(BigDecimal.valueOf(actualBaseline)).divide(BigDecimal.valueOf(actualBaseline), MathContext.DECIMAL64).toDouble() else null
                    val actualAbsDifference = if (value != null && actualBaseline != null) value.subtract(BigDecimal.valueOf(actualBaseline)) else null
                    if (actualBaseline != null && actualBaseline > 0.01 && actualDeviation != null && price != null && price > BigDecimal.ZERO && value != null) {
                        tempPortfolioRows.add(PortfolioRow(symbol, currency, quantity, price, value, actualBaseline, actualDeviation, actualAbsDifference, priceChange))
                    }
                }
            }
            validPortfolioItemsForTrading = tempPortfolioRows.sortedByDescending { it.deviation ?: Double.NEGATIVE_INFINITY }

            // Display and Financial Overview (uses loopCashBalance)
            if (validPortfolioItemsForTrading.isNotEmpty()) { /* ... Display Portfolio Summary ... */ }
            else if (loopCurrentHoldings.isNotEmpty()) { mainLoopLogger.info("ℹ️ No items with complete data for full portfolio summary display (e.g. missing prices or baselines).") }
            else if (loopCurrentHoldings.isEmpty()) { mainLoopLogger.info("ℹ️ Portfolio empty, no summary to display.") }

            mainLoopLogger.info("--- Financial Overview ---")
            mainLoopLogger.info("Total Holdings Value:   $QUOTE_CURRENCY_CODE ${totalHoldingsValue.setScale(2, RoundingMode.HALF_UP).toPlainString()}")
            mainLoopLogger.info("Cash Balance (Sim):     $QUOTE_CURRENCY_CODE ${loopCashBalance.setScale(2, RoundingMode.HALF_UP).toPlainString()}")
            val totalSimPortfolioValue = totalHoldingsValue.add(loopCashBalance) // Use simulated cash for this
            mainLoopLogger.info("Total Portfolio Value (Sim):  $QUOTE_CURRENCY_CODE ${totalSimPortfolioValue.setScale(2, RoundingMode.HALF_UP).toPlainString()}")

            // ... (Managed Deviation calculation using validPortfolioItemsForTrading as before) ...

            // ADZ & CP Logic (as before, using validPortfolioItemsForTrading)
            // ...

            // Trading Logic (as before, ensuring it uses loopCashBalance for decisions)
            // ...

            if (stateChangedThisCycle) { StateManager.saveState(botState) }
            val cycleEndTime = System.currentTimeMillis(); val elapsedMillis = cycleEndTime - cycleStartTime
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
