package org.bereft.adapter

import com.binance.api.client.domain.market.Candlestick as BinanceCandlestick
import org.bereft.MarketData
import vec.util.todub

/**
 * Adapt the Binance SDK's [BinanceCandlestick] into the internal [MarketData] abstraction.
 */
class BinanceMarketDataAdapter(private val source: BinanceCandlestick) : MarketData {
    override val openTime: Long get() = source.openTime ?: 0L
    override val open: Double get() = todub(source.open, Double.NaN)
    override val high: Double get() = todub(source.high, Double.NaN)
    override val low: Double get() = todub(source.low, Double.NaN)
    override val close: Double get() = todub(source.close, Double.NaN)
    override val volume: Double get() = todub(source.volume, Double.NaN)
    override val closeTime: Long get() = source.closeTime ?: 0L
}

/**
 * Convenience extension to invoke `asMarketData()` directly on a [BinanceCandlestick].
 */
fun BinanceCandlestick.asMarketData(): MarketData = BinanceMarketDataAdapter(this)