package com.example

import org.knowm.xchange.ExchangeFactory
import org.knowm.xchange.coinbasepro.CoinbaseProExchange
import org.knowm.xchange.currency.CurrencyPair
import org.knowm.xchange.service.marketdata.MarketDataService
import kotlin.math.*

class FusionTraderApp {
    // Market & Contract Specs
    companion object {
        const val INITIAL_PRICE = 50000.0        // BTC-USD
        const val VOLATILITY = 0.45              // 45% annual
        const val RISK_FREE = 0.03               // 3% annual
        const val STRIKE = INITIAL_PRICE
        const val DAYS_TO_EXP = 7
        const val SECONDS_PER_DAY = 24 * 60 * 60
        const val SECONDS_PER_YEAR = SECONDS_PER_DAY * 365.25
    }

    // Utility functions
    fun randn(): Double {
        var u = 0.0
        var v = 0.0
        while (u == 0.0) u = Math.random()
        while (v == 0.0) v = Math.random()
        return sqrt(-2 * ln(u)) * cos(2 * PI * v)
    }

    fun erf(x: Double): Double {
        // approximation of error function
        val a1 =  0.254829592
        val a2 = -0.284496736
        val a3 =  1.421413741
        val a4 = -1.453152027
        val a5 =  1.061405429
        val p  =  0.3275911

        val sign = if (x >= 0) 1 else -1
        val xAbs = abs(x)
        
        val t = 1.0 / (1.0 + p * xAbs)
        val y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-xAbs * xAbs)
        
        return sign * y
    }

    fun blackScholes(S: Double, K: Double, T: Double, sigma: Double, r: Double, call: Boolean = true): Double {
        val d1 = (ln(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T))
        val d2 = d1 - sigma * sqrt(T)
        val nd = { x: Double -> 0.5 * (1 + erf(x / sqrt(2))) }
        
        return if (call) S * nd(d1) - K * exp(-r * T) * nd(d2)
        else K * exp(-r * T) * nd(-d2) - S * nd(-d1)
    }

    fun delta(S: Double, K: Double, T: Double, sigma: Double, r: Double, call: Boolean = true): Double {
        val d1 = (ln(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T))
        val nd = { x: Double -> 0.5 * (1 + erf(x / sqrt(2))) }
        return if (call) nd(d1) else nd(d1) - 1
    }

    // Bot base class
    abstract class Bot(val name: String, val color: String, var cash: Double = 1e6) {
        var shares = 0.0
        var options = 0.0          // net position (+=long,-=short)
        var premium = 0.0          // total premium paid/received
        val log = mutableListOf<String>()
        var arenaDivId = ""
        val initialCash = cash

        fun logMsg(msg: String) {
            log.add(msg)
            if (log.size > 12) log.removeAt(0)
        }

        fun value(marketPrice: Double, strike: Double): Double {
            val intrinsicValue = if (options > 0) options * max(0.0, marketPrice - strike) else options * min(0.0, marketPrice - strike)
            return cash + shares * marketPrice + intrinsicValue
        }

        abstract fun think(market: Market)
    }

    // Delta Hedger Strategy
    class DeltaHedger(name: String, color: String) : Bot(name, color) {
        override fun think(market: Market) {
            val T = max(market.T, 1 / SECONDS_PER_YEAR)
            val d = market.fusionTraderApp.delta(market.price, STRIKE, T, VOLATILITY, RISK_FREE, true)
            val targetShares = -options * d
            val deltaShares = targetShares - shares
            
            if (abs(deltaShares) < 0.01) return
            
            if (deltaShares > 0) { // buy
                val qty = deltaShares
                val cost = qty * market.price
                if (cost <= cash) {
                    cash -= cost
                    shares += qty
                    logMsg("HEDGE BUY ${String.format("%.2f", qty)} @ ${String.format("%.0f", market.price)}")
                }
            } else { // sell
                val qty = -deltaShares
                cash += qty * market.price
                shares -= qty
                logMsg("HEDGE SELL ${String.format("%.2f", qty)} @ ${String.format("%.0f", market.price)}")
            }
        }
    }

    // Volatility Arbitrage Strategy
    class VolArb(name: String, color: String) : Bot(name, color) {
        private var targetVol = VOLATILITY
        
        override fun think(market: Market) {
            val realized = market.volatility()
            val T = max(market.T, 1 / SECONDS_PER_YEAR)
            val optPrice = market.fusionTraderApp.blackScholes(market.price, STRIKE, T, targetVol, RISK_FREE, true)
            val fairPrice = market.fusionTraderApp.blackScholes(market.price, STRIKE, T, realized, RISK_FREE, true)
            
            if (optPrice < fairPrice * 0.97) { // buy cheap
                val qty = 10.0
                if (cash > qty * optPrice) {
                    options += qty
                    cash -= qty * optPrice
                    premium += qty * optPrice
                    logMsg("VOL BUY 10 @ ${String.format("%.0f", optPrice)} (realVol ${String.format("%.1f", realized * 100)}%)")
                }
            } else if (optPrice > fairPrice * 1.03) { // sell rich
                val qty = 10.0
                options -= qty
                cash += qty * optPrice
                premium -= qty * optPrice
                logMsg("VOL SELL 10 @ ${String.format("%.0f", optPrice)} (realVol ${String.format("%.1f", realized * 100)}%)")
            }
        }
    }

    // Momentum Strategy
    class Momentum(name: String, color: String) : Bot(name, color) {
        override fun think(market: Market) {
            val historyIndex = max(0, market.history.size - 20)
            val ret = (market.price - market.history[historyIndex]) / market.history[historyIndex]
            
            if (ret > 0.02 && cash > 5000) {
                val qty = 1.0
                val T = max(market.T, 1 / SECONDS_PER_YEAR)
                val optPrice = market.fusionTraderApp.blackScholes(market.price, STRIKE, T, VOLATILITY, RISK_FREE, true)
                options += qty
                cash -= optPrice
                logMsg("MOMO CALL $qty @ ${String.format("%.0f", optPrice)}")
            }
            if (ret < -0.02 && options > 1) {
                val qty = 1.0
                val T = max(market.T, 1 / SECONDS_PER_YEAR)
                val optPrice = market.fusionTraderApp.blackScholes(market.price, STRIKE, T, VOLATILITY, RISK_FREE, true)
                options -= qty
                cash += optPrice
                logMsg("MOMO SELL $qty @ ${String.format("%.0f", optPrice)}")
            }
        }
    }

    // Market class
    inner class Market {
        var price = INITIAL_PRICE
        val history = mutableListOf<Double>()
        var T = DAYS_TO_EXP / 365.25
        var startTime = System.currentTimeMillis()
        var tick = 0
        val fusionTraderApp = this@FusionTraderApp

        init {
            history.add(INITIAL_PRICE)
        }

        fun volatility(): Double {
            if (history.size < 20) return VOLATILITY
            val returns = mutableListOf<Double>()
            for (i in 1 until 20) {
                val index = history.size - 20 + i
                val prevIndex = history.size - 20 + i - 1
                returns.add(ln(history[index] / history[prevIndex]))
            }
            val mean = returns.sum() / returns.size
            val varr = returns.map { r -> Math.pow(r - mean, 2.0) }.sum() / (returns.size - 1)
            return sqrt(varr * SECONDS_PER_YEAR) // annualized
        }

        fun step(bots: List<Bot>) {
            // 1. evolve price (geometric brownian)
            val dt = 1.0 / SECONDS_PER_DAY
            val drift = RISK_FREE - 0.5 * VOLATILITY * VOLATILITY
            val Z = randn()
            price *= exp(drift * dt + VOLATILITY * sqrt(dt) * Z)
            history.add(price)
            if (history.size > 200) history.removeAt(0)

            // 2. time decay
            T = max(0.0, (DAYS_TO_EXP * SECONDS_PER_DAY - (System.currentTimeMillis() - startTime) / 1000.0) / SECONDS_PER_YEAR)

            // 3. bots act
            bots.forEach { bot -> bot.think(this) }

            // 4. expiry
            if (T <= 0) {
                bots.forEach { bot ->
                    val settle = max(price - STRIKE, 0.0)
                    val payoff = bot.options * settle
                    bot.cash += payoff
                    bot.logMsg("EXPIRY settle ${String.format("%.0f", settle)} total payoff ${String.format("%.0f", payoff)}")
                    bot.options = 0.0
                    bot.shares = 0.0
                }
                T = DAYS_TO_EXP / 365.25
                startTime = System.currentTimeMillis()
            }
        }
    }

    fun startSimulation() {
        val bots = listOf(
            DeltaHedger("DeltaHedgeBot", "#0af"),
            VolArb("VolArbBot", "#f90"),
            Momentum("MomBot", "#3c3")
        )
        
        val market = Market()
        
        println("Fusion Trader Simulation Started")
        println("Initial Market Price: ${market.price}")
        println("Strike Price: $STRIKE")
        println("Days to Expiration: $DAYS_TO_EXP")
        
        // Run simulation for 100 ticks as a demonstration
        for (i in 0 until 100) {
            market.step(bots)
            if (i % 20 == 0) {
                println("\n--- Tick ${i} ---")
                println("Market Price: ${String.format("%.2f", market.price)}")
                println("Market T: ${String.format("%.4f", market.T)}")
                println("Market Vol: ${String.format("%.4f", market.volatility())}")
                
                bots.forEach { bot ->
                    println("${bot.name}: Cash=${String.format("%.0f", bot.cash)}, Shares=${String.format("%.2f", bot.shares)}, Options=${String.format("%.2f", bot.options)}, PnL=${String.format("%.0f", bot.value(market.price, STRIKE) - 1e6)}")
                }
            }
        }
        
        println("\n--- Final Results ---")
        bots.forEach { bot ->
            println("${bot.name}: Final PnL=${String.format("%.0f", bot.value(market.price, STRIKE) - 1e6)}")
        }
    }
}

