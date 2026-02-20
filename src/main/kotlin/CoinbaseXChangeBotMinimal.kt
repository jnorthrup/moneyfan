package com.example

import org.knowm.xchange.ExchangeFactory
import org.knowm.xchange.coinbasepro.CoinbaseProExchange
import org.knowm.xchange.currency.CurrencyPair
import org.knowm.xchange.service.marketdata.MarketDataService

fun main() {
    println("Starting CoinbaseXChangeBotMinimal...")

    // Initialize the CoinbasePro exchange
    val exchange = ExchangeFactory.INSTANCE.createExchange(CoinbaseProExchange::class.java)
    val marketDataService = exchange.marketDataService

    // Example: Print available trading pairs
    println("Available trading pairs:")
    val tradingPairs = marketDataService.fetchTickers()
    tradingPairs.forEach { println(it.key) }

    println("\nMinimal bot initialized successfully!")
}