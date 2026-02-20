package com.example

import io.javalin.Javalin
import io.javalin.http.Context
import org.knowm.xchange.ExchangeFactory
import org.knowm.xchange.coinbasepro.CoinbaseProExchange
import org.knowm.xchange.currency.CurrencyPair
import org.knowm.xchange.service.marketdata.MarketDataService
import java.util.*
import kotlin.math.*

// Import the FusionTraderApp classes
import com.example.FusionTraderApp.*

data class MarketData(
    val price: Double,
    val timestamp: Long,
    val volatility: Double
)

data class BotData(
    val name: String,
    val color: String,
    val cash: Double,
    val shares: Double,
    val options: Double,
    val pnl: Double,
    val log: List<String>
)

class FusionTraderServer {
    private val fusionTraderApp = FusionTraderApp()
    private val market = fusionTraderApp.Market()
    private val bots = listOf(
        fusionTraderApp.DeltaHedger("DeltaHedgeBot", "#0af"),
        fusionTraderApp.VolArb("VolArbBot", "#f90"),
        fusionTraderApp.Momentum("MomBot", "#3c3")
    )
    private var simulationRunning = false
    private var simulationThread: Thread? = null

    fun startServer(port: Int = 7000) {
        val app = Javalin.create { config ->
            config.staticFiles.add("/static") // Serve static files
        }.routes {
            it.get("/") { ctx -> serveFusionTrader(ctx) }
            it.get("/api/market-data") { ctx -> getMarketData(ctx) }
            it.get("/api/bot-data") { ctx -> getBotData(ctx) }
            it.post("/api/control/start") { ctx -> startSimulation(ctx) }
            it.post("/api/control/stop") { ctx -> stopSimulation(ctx) }
            it.post("/api/control/reset") { ctx -> resetSimulation(ctx) }
            it.get("/api/control/status") { ctx -> getStatus(ctx) }
        }

        app.start(port)
        println("Fusion Trader Server started on http://localhost:$port")
    }

    private fun serveFusionTrader(ctx: Context) {
        ctx.html(getFusionTraderHtml())
    }

    private fun getMarketData(ctx: Context) {
        val data = MarketData(
            price = market.price,
            timestamp = System.currentTimeMillis(),
            volatility = market.volatility()
        )
        ctx.json(data)
    }

    private fun getBotData(ctx: Context) {
        val botDataList = bots.map { bot ->
            val pnl = bot.value(market.price, FusionTraderApp.STRIKE) - bot.initialCash
            BotData(
                name = bot.name,
                color = bot.color,
                cash = bot.cash,
                shares = bot.shares,
                options = bot.options,
                pnl = pnl,
                log = bot.log.takeLast(10) // Last 10 log entries
            )
        }
        ctx.json(botDataList)
    }

    private fun startSimulation(ctx: Context) {
        if (!simulationRunning) {
            simulationRunning = true
            simulationThread = Thread {
                while (simulationRunning) {
                    try {
                        market.step(bots)
                        Thread.sleep(500) // Update every 500ms
                    } catch (e: InterruptedException) {
                        break
                    }
                }
            }
            simulationThread?.start()
        }
        ctx.json(mapOf("status" to "started"))
    }

    private fun stopSimulation(ctx: Context) {
        simulationRunning = false
        simulationThread?.interrupt()
        ctx.json(mapOf("status" to "stopped"))
    }

    private fun resetSimulation(ctx: Context) {
        simulationRunning = false
        simulationThread?.interrupt()
        
        // Reset market
        market.price = FusionTraderApp.INITIAL_PRICE
        market.history.clear()
        market.history.add(FusionTraderApp.INITIAL_PRICE)
        market.T = FusionTraderApp.DAYS_TO_EXP / 365.25
        market.startTime = System.currentTimeMillis()
        
        // Reset bots
        bots.forEach { bot ->
            bot.cash = bot.initialCash
            bot.shares = 0.0
            bot.options = 0.0
            bot.premium = 0.0
            bot.log.clear()
        }
        
        ctx.json(mapOf("status" to "reset"))
    }

    private fun getStatus(ctx: Context) {
        ctx.json(mapOf(
            "running" to simulationRunning,
            "marketPrice" to market.price,
            "timeToExpiration" to market.T * 365.25
        ))
    }

    private fun getFusionTraderHtml(): String {
        return """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>Fusion Trader - Advanced Crypto Options Arena</title>
            <style>
                body {
                    font-family: Arial, Helvetica, sans-serif;
                    background: #111;
                    color: #eee;
                    margin: 0;
                    padding: 0;
                    display: flex;
                    flex-direction: column;
                    height: 100vh;
                }
                header {
                    background: #000;
                    padding: .5rem 1rem;
                    font-size: 1.2rem;
                    font-weight: bold;
                    border-bottom: 2px solid #0af;
                }
                #arena {
                    flex: 1;
                    display: flex;
                    flex-wrap: wrap;
                    justify-content: space-around;
                    align-items: center;
                    padding: 1rem;
                    overflow-y: auto;
                }
                .bot {
                    width: 280px;
                    height: 220px;
                    border: 2px solid #0af;
                    border-radius: 8px;
                    background: #222;
                    display: flex;
                    flex-direction: column;
                    padding: .5rem;
                    margin: 0.5rem;
                }
                .bot h3 {
                    margin: 0 0 .3rem 0;
                    font-size: 1rem;
                    color: #0af;
                }
                .log {
                    flex: 1;
                    background: #000;
                    border-radius: 4px;
                    padding: .3rem;
                    font-size: .7rem;
                    overflow-y: auto;
                    white-space: pre-line;
                }
                #controls {
                    background: #000;
                    padding: .5rem 1rem;
                    display: flex;
                    gap: 1rem;
                    align-items: center;
                    border-top: 2px solid #0af;
                }
                button {
                    background: #0af;
                    border: none;
                    color: #000;
                    padding: .4rem .8rem;
                    font-weight: bold;
                    border-radius: 4px;
                    cursor: pointer;
                }
                button:hover {
                    background: #08c;
                }
                label {
                    font-size: .8rem;
                }
                input[type=range] {
                    width: 120px;
                }
                #market-info {
                    display: flex;
                    justify-content: space-around;
                    background: #000;
                    padding: 0.5rem;
                    border-top: 1px solid #555;
                }
                .market-data {
                    text-align: center;
                    margin: 0 1rem;
                }
                .market-data span {
                    display: block;
                    font-size: 0.9rem;
                }
                .market-data .value {
                    font-weight: bold;
                    color: #0af;
                }
                .pnl-positive { color: #0f0; }
                .pnl-negative { color: #f55; }
            </style>
        </head>
        <body>
            <header>Fusion Trader - Advanced Crypto Options Arena (Black-Scholes + Dynamic Hedging + Alpha Sniping)</header>
            
            <div id="market-info">
                <div class="market-data">
                    <span>Current Price</span>
                    <span id="current-price" class="value">50000.00</span>
                </div>
                <div class="market-data">
                    <span>Volatility</span>
                    <span id="current-vol" class="value">45.0%</span>
                </div>
                <div class="market-data">
                    <span>Time to Exp</span>
                    <span id="time-exp" class="value">7.00 days</span>
                </div>
                <div class="market-data">
                    <span>Total PnL</span>
                    <span id="total-pnl" class="value">0.00</span>
                </div>
            </div>
            
            <div id="arena"></div>
            
            <div id="controls">
                <button id="startBtn">Start Battle</button>
                <button id="resetBtn">Reset</button>
                <label>Update Speed <input type="range" id="speed" min="100" max="2000" value="500"></label>
                <button id="addBotBtn">Add Custom Bot</button>
            </div>

            <script>
                let simulationRunning = false;
                let updateInterval = null;
                
                async function fetchMarketData() {
                    try {
                        const response = await fetch('/api/market-data');
                        return await response.json();
                    } catch (error) {
                        console.error('Error fetching market data:', error);
                        return null;
                    }
                }
                
                async function fetchBotData() {
                    try {
                        const response = await fetch('/api/bot-data');
                        return await response.json();
                    } catch (error) {
                        console.error('Error fetching bot data:', error);
                        return [];
                    }
                }
                
                async function updateMarketInfo() {
                    const marketData = await fetchMarketData();
                    if (marketData) {
                        document.getElementById('current-price').textContent = marketData.price.toFixed(2);
                        document.getElementById('current-vol').textContent = (marketData.volatility * 100).toFixed(2) + '%';
                    }
                }
                
                async function updateBots() {
                    const botData = await fetchBotData();
                    
                    // Get all bot elements
                    const botElements = document.querySelectorAll('.bot');
                    
                    botData.forEach((bot, index) => {
                        if (index < botElements.length) {
                            const pnlClass = bot.pnl >= 0 ? 'pnl-positive' : 'pnl-negative';
                            
                            botElements[index].innerHTML = \`
                                <h3 style="color:\${bot.color}">\${bot.name}</h3>
                                <div class="log">Cash: \${bot.cash.toFixed(0)} USD
                                    Shares: \${bot.shares.toFixed(2)}
                                    Options: \${bot.options.toFixed(2)}
                                    PnL: <span class="\${pnlClass}">\${bot.pnl.toFixed(0)}</span>
                                    \${bot.log.join('\\n')}</div>\`;
                        }
                    });
                }
                
                async function startSimulation() {
                    try {
                        await fetch('/api/control/start', { method: 'POST' });
                        simulationRunning = true;
                        document.getElementById('startBtn').textContent = 'Pause';
                        
                        if (!updateInterval) {
                            updateInterval = setInterval(async () => {
                                if (simulationRunning) {
                                    await updateMarketInfo();
                                    await updateBots();
                                }
                            }, document.getElementById('speed').value);
                        }
                    } catch (error) {
                        console.error('Error starting simulation:', error);
                    }
                }
                
                async function stopSimulation() {
                    try {
                        await fetch('/api/control/stop', { method: 'POST' });
                        simulationRunning = false;
                        document.getElementById('startBtn').textContent = 'Start Battle';
                    } catch (error) {
                        console.error('Error stopping simulation:', error);
                    }
                }
                
                async function resetSimulation() {
                    try {
                        await fetch('/api/control/reset', { method: 'POST' });
                        simulationRunning = false;
                        document.getElementById('startBtn').textContent = 'Start Battle';
                        
                        clearInterval(updateInterval);
                        updateInterval = null;
                        
                        await updateMarketInfo();
                        await updateBots();
                    } catch (error) {
                        console.error('Error resetting simulation:', error);
                    }
                }
                
                // Initialize the app
                document.addEventListener('DOMContentLoaded', async () => {
                    // Initialize bot displays
                    await updateMarketInfo();
                    await updateBots();
                    
                    // Set up buttons
                    document.getElementById('startBtn').addEventListener('click', () => {
                        if (simulationRunning) {
                            stopSimulation();
                        } else {
                            startSimulation();
                        }
                    });
                    
                    document.getElementById('resetBtn').addEventListener('click', resetSimulation);
                    
                    document.getElementById('speed').addEventListener('input', () => {
                        if (updateInterval) {
                            clearInterval(updateInterval);
                            updateInterval = setInterval(async () => {
                                if (simulationRunning) {
                                    await updateMarketInfo();
                                    await updateBots();
                                }
                            }, document.getElementById('speed').value);
                        }
                    });
                    
                    document.getElementById('addBotBtn').addEventListener('click', () => {
                        alert('Adding bots dynamically is not supported in the server version. Add new bot types to the server code.');
                    });
                });
            </script>
        </body>
        </html>
        """
    }
}

// The main function for the server is defined in FusionTraderApp.kt