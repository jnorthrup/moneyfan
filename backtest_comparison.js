import initSqlJs from 'sql.js';
import fs from 'fs';
import path from 'path';

const DB_PATH = path.join(process.cwd(), 'backtest.db');

const COINBASE_PRODUCTS = [
    'BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD',
    'ADA-USD', 'DOGE-USD', 'AVAX-USD', 'DOT-USD',
    'LINK-USD', 'MATIC-USD', 'UNI-USD', 'ATOM-USD',
    'LTC-USD', 'BCH-USD', 'SHIB-USD', 'XLM-USD',
    'ALGO-USD', 'FIL-USD', 'AAVE-USD', 'NEAR-USD',
    'APT-USD', 'ARB-USD', 'OP-USD', 'INJ-USD',
    'SUI-USD', 'SEI-USD', 'TIA-USD', 'WLD-USD',
    'PEPE-USD', 'FET-USD'
];

const CIRCULATORY_PAIRS = [
    'ETH-BTC', 'SOL-BTC', 'XRP-BTC', 'ADA-BTC',
    'DOGE-BTC', 'AVAX-BTC', 'DOT-BTC', 'LINK-BTC',
    'MATIC-BTC', 'UNI-BTC', 'ATOM-BTC', 'LTC-BTC',
    'ALGO-BTC', 'FIL-BTC', 'NEAR-BTC'
];

const SYMBOL_MAP = {
    'BTC-USD': 'BTC', 'ETH-USD': 'ETH', 'SOL-USD': 'SOL', 'XRP-USD': 'XRP',
    'ADA-USD': 'ADA', 'DOGE-USD': 'DOGE', 'AVAX-USD': 'AVAX', 'DOT-USD': 'DOT',
    'LINK-USD': 'LINK', 'MATIC-USD': 'MATIC', 'UNI-USD': 'UNI', 'ATOM-USD': 'ATOM',
    'LTC-USD': 'LTC', 'BCH-USD': 'BCH', 'SHIB-USD': 'SHIB', 'XLM-USD': 'XLM',
    'ALGO-USD': 'ALGO', 'FIL-USD': 'FIL', 'AAVE-USD': 'AAVE', 'NEAR-USD': 'NEAR',
    'APT-USD': 'APT', 'ARB-USD': 'ARB', 'OP-USD': 'OP', 'INJ-USD': 'INJ',
    'SUI-USD': 'SUI', 'SEI-USD': 'SEI', 'TIA-USD': 'TIA', 'WLD-USD': 'WLD',
    'PEPE-USD': 'PEPE', 'FET-USD': 'FET',
    'ETH-BTC': 'ETH/BTC', 'SOL-BTC': 'SOL/BTC', 'XRP-BTC': 'XRP/BTC', 'ADA-BTC': 'ADA/BTC',
    'DOGE-BTC': 'DOGE/BTC', 'AVAX-BTC': 'AVAX/BTC', 'DOT-BTC': 'DOT/BTC', 'LINK-BTC': 'LINK/BTC',
    'MATIC-BTC': 'MATIC/BTC', 'UNI-BTC': 'UNI/BTC', 'ATOM-BTC': 'ATOM/BTC', 'LTC-BTC': 'LTC/BTC',
    'ALGO-BTC': 'ALGO/BTC', 'FIL-BTC': 'FIL/BTC', 'NEAR-BTC': 'NEAR/BTC'
};

async function fetchCoinbaseCandles(productId, days = 7, granularity = 300) {
    const now = Math.floor(Date.now() / 1000);
    const startTs = now - (days * 24 * 60 * 60);
    const maxCandles = 300;
    const chunkSize = maxCandles * granularity;
    
    let allCandles = [];
    let chunkStart = startTs;
    let chunkCount = 0;
    const totalChunks = Math.ceil((now - startTs) / chunkSize);
    
    while (chunkStart < now) {
        const chunkEnd = Math.min(chunkStart + chunkSize, now);
        const url = `https://api.exchange.coinbase.com/products/${productId}/candles?granularity=${granularity}&start=${chunkStart}&end=${chunkEnd}`;
        
        const response = await fetch(url);
        if (!response.ok) break;
        
        const data = await response.json();
        if (Array.isArray(data) && data.length > 0) {
            allCandles = allCandles.concat(data);
        }
        
        chunkStart = chunkEnd;
        chunkCount++;
        if (chunkCount % 10 === 0) {
            process.stdout.write(`\r    Chunk ${chunkCount}/${totalChunks}, ${allCandles.length} candles`);
        }
        await new Promise(r => setTimeout(r, 100));
    }
    
    return allCandles.map(c => ({
        time: c[0],
        low: c[1],
        high: c[2],
        open: c[3],
        close: c[4],
        volume: c[5]
    })).sort((a, b) => a.time - b.time);
}

class BacktestDB {
    constructor() {
        this.db = null;
    }
    
    async init() {
        const SQL = await initSqlJs();
        if (fs.existsSync(DB_PATH)) {
            const buffer = fs.readFileSync(DB_PATH);
            this.db = new SQL.Database(buffer);
        } else {
            this.db = new SQL.Database();
        }
        
        this.db.run(`
            CREATE TABLE IF NOT EXISTS candles (
                symbol TEXT,
                exchange TEXT,
                time INTEGER,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                PRIMARY KEY (symbol, exchange, time)
            )
        `);
        this.save();
    }
    
    save() {
        const data = this.db.export();
        fs.writeFileSync(DB_PATH, Buffer.from(data));
    }
    
    run(sql, params = []) { this.db.run(sql, params); }
    
    all(sql, params = []) {
        const stmt = this.db.prepare(sql);
        stmt.bind(params);
        const results = [];
        while (stmt.step()) results.push(stmt.getAsObject());
        stmt.free();
        return results;
    }
    
    get(sql, params = []) {
        const results = this.all(sql, params);
        return results[0] || null;
    }
    
    async loadData(days = 730, granularity = 3600) {
        const granName = granularity === 60 ? '1min' : granularity === 300 ? '5min' : granularity === 3600 ? '1hr' : `${granularity}s`;
        console.log(`📥 Fetching ${days} days (${granName}) from Coinbase...`);
        console.log(`   Expected: ~${Math.floor(days * 24 * 60 * 60 / granularity)} candles per asset`);
        
        const totalAssets = COINBASE_PRODUCTS.length + CIRCULATORY_PAIRS.length;
        let assetIndex = 0;
        
        for (const productId of COINBASE_PRODUCTS) {
            assetIndex++;
            const symbol = SYMBOL_MAP[productId];
            process.stdout.write(`\r  [${assetIndex}/${totalAssets}] ${symbol}: fetching...`);
            const candles = await fetchCoinbaseCandles(productId, days, granularity);
            process.stdout.write(`\r  [${assetIndex}/${totalAssets}] ${symbol}: ${candles.length} candles    \n`);
            for (const c of candles) {
                this.db.run(`INSERT OR REPLACE INTO candles (symbol, exchange, time, open, high, low, close, volume) VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
                    [symbol, 'coinbase', c.time, c.open, c.high, c.low, c.close, c.volume]);
            }
            this.save();
        }
        
        console.log(`📥 Fetching circulatory pairs (ETH-BTC, etc)...`);
        for (const pair of CIRCULATORY_PAIRS) {
            assetIndex++;
            const symbol = SYMBOL_MAP[pair];
            process.stdout.write(`\r  [${assetIndex}/${totalAssets}] ${symbol}: fetching...`);
            const candles = await fetchCoinbaseCandles(pair, days, granularity);
            process.stdout.write(`\r  [${assetIndex}/${totalAssets}] ${symbol}: ${candles.length} candles    \n`);
            for (const c of candles) {
                this.db.run(`INSERT OR REPLACE INTO candles (symbol, exchange, time, open, high, low, close, volume) VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
                    [symbol, 'coinbase', c.time, c.open, c.high, c.low, c.close, c.volume]);
            }
            this.save();
        }
        
        console.log(`\n✅ Data load complete`);
    }
    
    getPriceAt(symbol, time) {
        const result = this.get(`SELECT close FROM candles WHERE symbol = ? AND exchange = 'coinbase' AND time <= ? ORDER BY time DESC LIMIT 1`,
            [symbol, Math.floor(time / 1000)]);
        return result?.close || null;
    }
    
    getTimeRange() {
        return this.get(`SELECT MIN(time) as start_time, MAX(time) as end_time FROM candles`);
    }
    
    close() { this.save(); this.db.close(); }
}

// ============================================
// AUTHOR'S STRATEGY (from cryptobot-kraken-baseline.js)
// ============================================
const AUTHOR_CONFIG = {
    name: "Author's Original",
    HARVEST_EXCLUDE: ["BTC", "USDC", "USD"],
    REBALANCE_EXCLUDE: ["BTC", "USDC", "USD"],
    TARGET_ADJUST_PERCENT: 0.000,
    FLAT_HARVEST_TRIGGER_PERCENT: 0.03,
    HARVEST_CYCLE_THRESHOLD: 3,
    MIN_SURPLUS_FOR_HARVEST: 1.00,
    FLAT_REBALANCE_TRIGGER_PERCENT: 0.04,
    PARTIAL_RECOVERY_PERCENT: 0.875,
    REBALANCE_POSITIVE_THRESHOLD: 3,
    HARVEST_ALLOC_BTC_PERCENT: 0.10,
    HARVEST_ALLOC_REINVEST_PERCENT: 0.50,
    HARVEST_ALLOC_CASH_PERCENT: 0.40,
    ENABLE_ADAPTIVE_DEAD_ZONE: true,
    ADAPTIVE_DZ_HARVEST_TRIGGER_PERCENT: 0.020,
    ADAPTIVE_DZ_REBALANCE_TRIGGER_PERCENT: 0.020,
    ENABLE_CRASH_PROTECTION: true,
    CP_TRIGGER_ASSET_PERCENT: 0.70
};

// ============================================
// KILO'S SUGGESTIONS (alternative parameters)
// ============================================
const KILO_CONFIG = {
    name: "Kilo's Suggestion",
    HARVEST_EXCLUDE: ["BTC"],
    REBALANCE_EXCLUDE: ["BTC"],
    TARGET_ADJUST_PERCENT: 0.005,
    FLAT_HARVEST_TRIGGER_PERCENT: {
        'BTC': 0.02, 'ETH': 0.03, 'SOL': 0.05, 'XRP': 0.04,
        'ADA': 0.05, 'DOGE': 0.08, 'AVAX': 0.06, 'DOT': 0.05,
        'LINK': 0.04, 'MATIC': 0.05, 'UNI': 0.04, 'ATOM': 0.05,
        'LTC': 0.03, 'BCH': 0.04, 'SHIB': 0.10, 'XLM': 0.05,
        'ALGO': 0.05, 'FIL': 0.06, 'AAVE': 0.05, 'NEAR': 0.05,
        'APT': 0.07, 'ARB': 0.06, 'OP': 0.06, 'INJ': 0.06,
        'SUI': 0.08, 'SEI': 0.08, 'TIA': 0.07, 'WLD': 0.10,
        'PEPE': 0.12, 'FET': 0.08,
        'ETH/BTC': 0.04, 'SOL/BTC': 0.06, 'XRP/BTC': 0.05,
        'ADA/BTC': 0.06, 'LINK/BTC': 0.05, 'DOT/BTC': 0.06
    },
    HARVEST_CYCLE_THRESHOLD: 2,
    MIN_SURPLUS_FOR_HARVEST: 10.00,
    FLAT_REBALANCE_TRIGGER_PERCENT: {
        'BTC': 0.02, 'ETH': 0.03, 'SOL': 0.05, 'XRP': 0.04,
        'ADA': 0.05, 'DOGE': 0.08, 'AVAX': 0.06, 'DOT': 0.05,
        'LINK': 0.04, 'MATIC': 0.05, 'UNI': 0.04, 'ATOM': 0.05,
        'LTC': 0.03, 'BCH': 0.04, 'SHIB': 0.10, 'XLM': 0.05,
        'ALGO': 0.05, 'FIL': 0.06, 'AAVE': 0.05, 'NEAR': 0.05,
        'APT': 0.07, 'ARB': 0.06, 'OP': 0.06, 'INJ': 0.06,
        'SUI': 0.08, 'SEI': 0.08, 'TIA': 0.07, 'WLD': 0.10,
        'PEPE': 0.12, 'FET': 0.08,
        'ETH/BTC': 0.04, 'SOL/BTC': 0.06, 'XRP/BTC': 0.05,
        'ADA/BTC': 0.06, 'LINK/BTC': 0.05, 'DOT/BTC': 0.06
    },
    PARTIAL_RECOVERY_PERCENT: 0.75,
    REBALANCE_POSITIVE_THRESHOLD: 2,
    HARVEST_ALLOC_BTC_PERCENT: 0.20,
    HARVEST_ALLOC_REINVEST_PERCENT: 0.60,
    HARVEST_ALLOC_CASH_PERCENT: 0.20,
    ENABLE_ADAPTIVE_DEAD_ZONE: false,
    ADAPTIVE_DZ_HARVEST_TRIGGER_PERCENT: 0.015,
    ADAPTIVE_DZ_REBALANCE_TRIGGER_PERCENT: 0.015,
    ENABLE_CRASH_PROTECTION: true,
    CP_TRIGGER_ASSET_PERCENT: 0.50
};

class StrategyBacktester {
    constructor(db, config, initialHoldings, initialPrices) {
        this.db = db;
        this.config = config;
        this.name = config.name;
        
        this.cash = 5000;
        this.holdings = new Map(Object.entries(initialHoldings));
        this.baselines = new Map();
        this.trades = [];
        this.harvestCycles = new Map();
        this.rebalanceCycles = new Map();
        this.inAdaptiveZone = new Map();
        
        for (const [symbol, qty] of Object.entries(initialHoldings)) {
            const price = initialPrices[symbol] || 100;
            this.baselines.set(symbol, qty * price);
        }
    }
    
    getHarvestTrigger(symbol) {
        if (typeof this.config.FLAT_HARVEST_TRIGGER_PERCENT === 'object') {
            return this.config.FLAT_HARVEST_TRIGGER_PERCENT[symbol] || 0.03;
        }
        return this.config.FLAT_HARVEST_TRIGGER_PERCENT;
    }
    
    getRebalanceTrigger(symbol) {
        if (typeof this.config.FLAT_REBALANCE_TRIGGER_PERCENT === 'object') {
            return this.config.FLAT_REBALANCE_TRIGGER_PERCENT[symbol] || 0.04;
        }
        return this.config.FLAT_REBALANCE_TRIGGER_PERCENT;
    }
    
    run() {
        const timeRange = this.db.getTimeRange();
        if (!timeRange.start_time) return null;
        
        const startTime = timeRange.start_time * 1000;
        const endTime = timeRange.end_time * 1000;
        const interval = 5 * 60 * 1000;
        
        let currentTime = startTime;
        let peakValue = 0;
        let maxDrawdown = 0;
        
        while (currentTime <= endTime) {
            const portfolioValue = this._getPortfolioValue(currentTime);
            if (portfolioValue > peakValue) peakValue = portfolioValue;
            const drawdown = (peakValue - portfolioValue) / peakValue;
            if (drawdown > maxDrawdown) maxDrawdown = drawdown;
            
            this._runCycle(currentTime);
            currentTime += interval;
        }
        
        return {
            name: this.name,
            finalValue: this._getPortfolioValue(endTime),
            cash: this.cash,
            holdings: Object.fromEntries(this.holdings),
            tradeCount: this.trades.length,
            peakValue,
            maxDrawdown,
            trades: this.trades.slice(-5)
        };
    }
    
    _getPortfolioValue(timestamp) {
        let total = this.cash;
        for (const [symbol, qty] of this.holdings.entries()) {
            const price = this.db.getPriceAt(symbol, timestamp);
            if (price) total += qty * price;
        }
        return total;
    }
    
    _runCycle(timestamp) {
        const prices = {};
        for (const symbol of this.holdings.keys()) {
            prices[symbol] = this.db.getPriceAt(symbol, timestamp);
        }
        
        for (const [symbol, qty] of this.holdings.entries()) {
            if (this.config.HARVEST_EXCLUDE.includes(symbol)) continue;
            
            const price = prices[symbol];
            if (!price) continue;
            
            const value = qty * price;
            const baseline = this.baselines.get(symbol) || value;
            const deviation = (value - baseline) / baseline;
            const harvestTrigger = this.getHarvestTrigger(symbol);
            
            if (!this.harvestCycles.has(symbol)) this.harvestCycles.set(symbol, 0);
            
            if (deviation >= harvestTrigger) {
                this.harvestCycles.set(symbol, this.harvestCycles.get(symbol) + 1);
                
                if (this.harvestCycles.get(symbol) >= this.config.HARVEST_CYCLE_THRESHOLD) {
                    const surplus = value - baseline;
                    if (surplus >= this.config.MIN_SURPLUS_FOR_HARVEST) {
                        const sellQty = surplus / price;
                        if (sellQty < qty) {
                            this.holdings.set(symbol, qty - sellQty);
                            const sellValue = sellQty * price;
                            const fee = sellValue * 0.004;
                            this.cash += sellValue - fee;
                            this.baselines.set(symbol, baseline * (1 + this.config.TARGET_ADJUST_PERCENT));
                            this.trades.push({ symbol, side: 'SELL', qty: sellQty, price, value: sellValue });
                        }
                    }
                    this.harvestCycles.set(symbol, 0);
                }
            } else {
                this.harvestCycles.set(symbol, 0);
            }
            
            if (!this.rebalanceCycles.has(symbol)) this.rebalanceCycles.set(symbol, 0);
            
            const rebalanceTrigger = this.getRebalanceTrigger(symbol);
            if (deviation <= -rebalanceTrigger && !this.config.REBALANCE_EXCLUDE.includes(symbol)) {
                this.rebalanceCycles.set(symbol, this.rebalanceCycles.get(symbol) + 1);
                
                if (this.rebalanceCycles.get(symbol) >= this.config.REBALANCE_POSITIVE_THRESHOLD) {
                    const shortfall = baseline - value;
                    const buyAmount = shortfall * this.config.PARTIAL_RECOVERY_PERCENT;
                    if (buyAmount >= 1 && this.cash >= buyAmount) {
                        const buyQty = buyAmount / price;
                        this.holdings.set(symbol, (this.holdings.get(symbol) || 0) + buyQty);
                        const fee = buyAmount * 0.006;
                        this.cash -= buyAmount + fee;
                        this.baselines.set(symbol, baseline * (1 - this.config.TARGET_ADJUST_PERCENT));
                        this.trades.push({ symbol, side: 'BUY', qty: buyQty, price, value: buyAmount });
                    }
                    this.rebalanceCycles.set(symbol, 0);
                }
            } else {
                this.rebalanceCycles.set(symbol, 0);
            }
        }
    }
}

// ============================================
// SOTA STRATEGY: Momentum with Trailing Stop
// Based on trend-following + volatility-adjusted position sizing
// ============================================
const SOTA_CONFIG = {
    name: "Momentum-Trailing",
    LOOKBACK_PERIOD: 20,
    MOMENTUM_THRESHOLD: 0.015,
    TRAILING_STOP_PERCENT: 0.12,
    VOLATILITY_LOOKBACK: 14,
    MAX_POSITION_PERCENT: 0.35,
    REBALANCE_INTERVAL: 24 * 60 * 60 * 1000,
    RISK_PER_TRADE: 0.02,
    MIN_TRADE_SIZE: 0.01
};

class MomentumBacktester {
    constructor(db, config, initialHoldings, initialPrices) {
        this.db = db;
        this.config = config;
        this.name = config.name;
        
        this.cash = 2500;
        this.holdings = new Map(Object.entries(initialHoldings));
        this.trades = [];
        this.priceHistory = new Map();
        this.trailingStops = new Map();
        this.lastRebalance = 0;
        this.peakValues = new Map();
        
        for (const symbol of Object.keys(initialHoldings)) {
            this.priceHistory.set(symbol, []);
            this.peakValues.set(symbol, 0);
        }
    }
    
    run() {
        const timeRange = this.db.getTimeRange();
        if (!timeRange.start_time) return null;
        
        const startTime = timeRange.start_time * 1000;
        const endTime = timeRange.end_time * 1000;
        const interval = 5 * 60 * 1000;
        
        let currentTime = startTime;
        let peakValue = 0;
        let maxDrawdown = 0;
        
        while (currentTime <= endTime) {
            const portfolioValue = this._getPortfolioValue(currentTime);
            if (portfolioValue > peakValue) peakValue = portfolioValue;
            const drawdown = (peakValue - portfolioValue) / peakValue;
            if (drawdown > maxDrawdown) maxDrawdown = drawdown;
            
            this._runCycle(currentTime);
            currentTime += interval;
        }
        
        return {
            name: this.name,
            finalValue: this._getPortfolioValue(endTime),
            cash: this.cash,
            holdings: Object.fromEntries(this.holdings),
            tradeCount: this.trades.length,
            peakValue,
            maxDrawdown,
            trades: this.trades.slice(-5)
        };
    }
    
    _getPortfolioValue(timestamp) {
        let total = this.cash;
        for (const [symbol, qty] of this.holdings.entries()) {
            const price = this.db.getPriceAt(symbol, timestamp);
            if (price) total += qty * price;
        }
        return total;
    }
    
    _runCycle(timestamp) {
        const prices = {};
        const symbols = Object.keys(this.holdings);
        
        for (const symbol of symbols) {
            prices[symbol] = this.db.getPriceAt(symbol, timestamp);
            if (prices[symbol]) {
                const history = this.priceHistory.get(symbol) || [];
                history.push({ time: timestamp, price: prices[symbol] });
                if (history.length > this.config.LOOKBACK_PERIOD * 2) history.shift();
                this.priceHistory.set(symbol, history);
            }
        }
        
        for (const [symbol, qty] of this.holdings.entries()) {
            const price = prices[symbol];
            if (!price) continue;
            
            const history = this.priceHistory.get(symbol) || [];
            if (history.length < this.config.LOOKBACK_PERIOD) continue;
            
            const momentum = this._calculateMomentum(history);
            const volatility = this._calculateVolatility(history);
            const atr = this._calculateATR(history);
            
            const peak = this.peakValues.get(symbol) || price;
            if (price > peak) {
                this.peakValues.set(symbol, price);
                this.trailingStops.set(symbol, price * (1 - this.config.TRAILING_STOP_PERCENT));
            }
            
            const trailingStop = this.trailingStops.get(symbol);
            if (trailingStop && price < trailingStop && qty > 0.01) {
                const sellQty = qty * 0.25;
                if (sellQty > 0.001) {
                    const sellValue = sellQty * price;
                    const fee = sellValue * 0.004;
                    this.cash += sellValue - fee;
                    this.holdings.set(symbol, qty - sellQty);
                    this.trades.push({ symbol, side: 'SELL', qty: sellQty, price, value: sellValue, reason: 'TrailingStop' });
                }
                this.peakValues.set(symbol, price);
                this.trailingStops.set(symbol, price * (1 - this.config.TRAILING_STOP_PERCENT));
            }
            
            if (momentum < -this.config.MOMENTUM_THRESHOLD && qty > 0.01) {
                const sellRatio = Math.min(0.3, Math.abs(momentum));
                const sellQty = qty * sellRatio;
                if (sellQty > 0.001) {
                    const sellValue = sellQty * price;
                    const fee = sellValue * 0.004;
                    this.cash += sellValue - fee;
                    this.holdings.set(symbol, qty - sellQty);
                    this.trades.push({ symbol, side: 'SELL', qty: sellQty, price, value: sellValue, reason: 'MomentumExit' });
                }
            }
        }
        
        if (timestamp - this.lastRebalance >= this.config.REBALANCE_INTERVAL) {
            this._rebalance(prices, timestamp);
            this.lastRebalance = timestamp;
        }
    }
    
    _calculateMomentum(history) {
        if (history.length < this.config.LOOKBACK_PERIOD) return 0;
        const recent = history.slice(-this.config.LOOKBACK_PERIOD);
        const startPrice = recent[0].price;
        const endPrice = recent[recent.length - 1].price;
        return (endPrice - startPrice) / startPrice;
    }
    
    _calculateVolatility(history) {
        if (history.length < 2) return 0;
        const returns = [];
        for (let i = 1; i < history.length; i++) {
            returns.push((history[i].price - history[i-1].price) / history[i-1].price);
        }
        const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
        return Math.sqrt(returns.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / returns.length);
    }
    
    _calculateATR(history) {
        if (history.length < 2) return 0;
        let trSum = 0;
        for (let i = 1; i < history.length; i++) {
            trSum += Math.abs(history[i].price - history[i-1].price);
        }
        return trSum / (history.length - 1);
    }
    
    _rebalance(prices, timestamp) {
        const portfolioValue = this._getPortfolioValue(timestamp);
        const targetAllocation = portfolioValue * this.config.MAX_POSITION_PERCENT;
        const symbols = Object.keys(this.holdings).filter(s => s !== 'BTC');
        
        for (const symbol of symbols) {
            const history = this.priceHistory.get(symbol) || [];
            if (history.length < this.config.LOOKBACK_PERIOD) continue;
            
            const momentum = this._calculateMomentum(history);
            const price = prices[symbol];
            if (!price) continue;
            
            const currentValue = (this.holdings.get(symbol) || 0) * price;
            
            if (momentum > this.config.MOMENTUM_THRESHOLD && currentValue < targetAllocation) {
                const buyAmount = Math.min(targetAllocation * 0.1, this.cash * 0.1);
                if (buyAmount >= 5) {
                    const buyQty = buyAmount / price;
                    const fee = buyAmount * 0.006;
                    this.holdings.set(symbol, (this.holdings.get(symbol) || 0) + buyQty);
                    this.cash -= buyAmount + fee;
                    this.trades.push({ symbol, side: 'BUY', qty: buyQty, price, value: buyAmount, reason: 'MomentumEntry' });
                }
            }
        }
    }
}

async function main() {
    console.log('═══════════════════════════════════════════════════════');
    console.log('       STRATEGY COMPARISON BACKTEST');
    console.log('═══════════════════════════════════════════════════════\n');
    
    const db = new BacktestDB();
    await db.init();
    
    const count = db.all(`SELECT COUNT(*) as cnt FROM candles`);
    const totalCandles = count[0]?.cnt || 0;
    
    const NEED_2YR = 30 * 2 * 365 * 24;
    
    if (totalCandles < NEED_2YR) {
        console.log(`Current candles: ${totalCandles.toLocaleString()}`);
        console.log(`Target: ${NEED_2YR.toLocaleString()} (30 assets × 2 years × hourly)\n`);
        await db.loadData(730, 3600);
    } else {
        console.log(`✅ Data already loaded: ${totalCandles.toLocaleString()} candles\n`);
    }
    
    const timeRange = db.getTimeRange();
    console.log(`📅 Period: ${new Date(timeRange.start_time * 1000).toISOString().slice(0,10)} to ${new Date(timeRange.end_time * 1000).toISOString().slice(0,10)}\n`);
    
    const initialHoldings = {
        'BTC': 0.03, 'ETH': 0.75, 'SOL': 12, 'XRP': 350,
        'ADA': 1000, 'DOGE': 2000, 'AVAX': 8, 'DOT': 30,
        'LINK': 12, 'MATIC': 150, 'UNI': 15, 'ATOM': 20,
        'LTC': 0.4, 'BCH': 0.5, 'SHIB': 500000, 'XLM': 1500,
        'ALGO': 200, 'FIL': 15, 'AAVE': 0.5, 'NEAR': 25,
        'APT': 10, 'ARB': 25, 'OP': 20, 'INJ': 3,
        'SUI': 50, 'SEI': 100, 'TIA': 15, 'WLD': 25,
        'PEPE': 5000000, 'FET': 15,
        'ETH/BTC': 15, 'SOL/BTC': 150, 'XRP/BTC': 4000,
        'ADA/BTC': 12000, 'LINK/BTC': 150, 'DOT/BTC': 400
    };
    const initialPrices = {
        'BTC': 68000, 'ETH': 2000, 'SOL': 80, 'XRP': 1.40,
        'ADA': 0.35, 'DOGE': 0.08, 'AVAX': 25, 'DOT': 4,
        'LINK': 12, 'MATIC': 0.50, 'UNI': 6, 'ATOM': 5,
        'LTC': 70, 'BCH': 350, 'SHIB': 0.00001, 'XLM': 0.10,
        'ALGO': 0.15, 'FIL': 4, 'AAVE': 80, 'NEAR': 4,
        'APT': 8, 'ARB': 0.60, 'OP': 1.50, 'INJ': 25,
        'SUI': 1.20, 'SEI': 0.30, 'TIA': 8, 'WLD': 2,
        'PEPE': 0.0000008, 'FET': 1.20,
        'ETH/BTC': 0.0294, 'SOL/BTC': 0.00118,
        'XRP/BTC': 0.0000206, 'ADA/BTC': 0.0000051,
        'LINK/BTC': 0.000176, 'DOT/BTC': 0.000059
    };
    
    const authorBacktester = new StrategyBacktester(db, AUTHOR_CONFIG, initialHoldings, initialPrices);
    const kiloBacktester = new StrategyBacktester(db, KILO_CONFIG, initialHoldings, initialPrices);
    const sotaBacktester = new MomentumBacktester(db, SOTA_CONFIG, initialHoldings, initialPrices);
    
    const authorResult = authorBacktester.run();
    const kiloResult = kiloBacktester.run();
    const sotaResult = sotaBacktester.run();
    
    console.log('┌─────────────────────────────────────────────────────────────────┐');
    console.log('│                         RESULTS                                │');
    console.log('├─────────────────────────────────────────────────────────────────┤');
    console.log('│ Metric          │ Author        │ Kilo          │ SOTA         │');
    console.log('├─────────────────────────────────────────────────────────────────┤');
    console.log(`│ Final Value     │ $${authorResult.finalValue.toFixed(2).padEnd(11)} │ $${kiloResult.finalValue.toFixed(2).padEnd(11)} │ $${sotaResult.finalValue.toFixed(2).padEnd(10)} │`);
    console.log(`│ Cash            │ $${authorResult.cash.toFixed(2).padEnd(11)} │ $${kiloResult.cash.toFixed(2).padEnd(11)} │ $${sotaResult.cash.toFixed(2).padEnd(10)} │`);
    console.log(`│ Trades          │ ${authorResult.tradeCount.toString().padEnd(13)} │ ${kiloResult.tradeCount.toString().padEnd(13)} │ ${sotaResult.tradeCount.toString().padEnd(12)} │`);
    console.log(`│ Max Drawdown    │ ${(authorResult.maxDrawdown * 100).toFixed(1)}%`.padEnd(17) + `│ ${(kiloResult.maxDrawdown * 100).toFixed(1)}%`.padEnd(15) + `│ ${(sotaResult.maxDrawdown * 100).toFixed(1)}%`.padEnd(13) + '│');
    console.log('└─────────────────────────────────────────────────────────────────┘\n');
    
    const best = authorResult.finalValue >= kiloResult.finalValue && authorResult.finalValue >= sotaResult.finalValue ? 'Author' :
                 kiloResult.finalValue >= sotaResult.finalValue ? 'Kilo' : 'SOTA';
    const bestValue = Math.max(authorResult.finalValue, kiloResult.finalValue, sotaResult.finalValue);
    console.log(`🏆 Winner: ${best} ($${bestValue.toFixed(2)})\n`);
    
    console.log('═══════════════════════════════════════════════════════');
    console.log('                 CONFIG DIFFERENCES');
    console.log('═══════════════════════════════════════════════════════\n');
    
    console.log('┌─────────────────────────────────────────────────────┐');
    console.log('│ Parameter            │ Author    │ Kilo      │');
    console.log('├─────────────────────────────────────────────────────┤');
    console.log(`│ Harvest Trigger      │ 3.0%      │ Per-asset │`);
    console.log(`│   - BTC              │ 3.0%      │ 2.0%      │`);
    console.log(`│   - ETH              │ 3.0%      │ 3.0%      │`);
    console.log(`│   - SOL              │ 3.0%      │ 5.0%      │`);
    console.log(`│   - XRP              │ 3.0%      │ 4.0%      │`);
    console.log(`│ Rebalance Trigger    │ 4.0%      │ Per-asset │`);
    console.log(`│ Harvest Cycles       │ 3         │ 2         │`);
    console.log(`│ Min Harvest          │ $1        │ $10       │`);
    console.log(`│ Target Adjust        │ 0.0%      │ 0.5%      │`);
    console.log(`│ BTC Alloc            │ 10%       │ 20%       │`);
    console.log(`│ Reinvest Alloc       │ 50%       │ 60%       │`);
    console.log(`│ Adaptive Dead Zone   │ ON        │ OFF       │`);
    console.log('└─────────────────────────────────────────────────────┘\n');
    
    console.log('Author last trades:');
    for (const t of authorResult.trades) {
        console.log(`  ${t.side} ${t.qty.toFixed(6)} ${t.symbol} @ $${t.price.toFixed(2)}`);
    }
    
    console.log('\nKilo last trades:');
    for (const t of kiloResult.trades) {
        console.log(`  ${t.side} ${t.qty.toFixed(6)} ${t.symbol} @ $${t.price.toFixed(2)}`);
    }
    
    console.log('\nSOTA last trades:');
    for (const t of sotaResult.trades) {
        const reason = t.reason ? ` [${t.reason}]` : '';
        console.log(`  ${t.side} ${t.qty.toFixed(6)} ${t.symbol} @ $${t.price.toFixed(2)}${reason}`);
    }
    
    console.log('\n═══════════════════════════════════════════════════════');
    console.log('              SOTA STRATEGY CONFIG');
    console.log('═══════════════════════════════════════════════════════');
    console.log('┌─────────────────────────────────────────────────────┐');
    console.log('│ Parameter            │ Value      │');
    console.log('├─────────────────────────────────────────────────────┤');
    console.log('│ Lookback Period      │ 20 candles │');
    console.log('│ Momentum Threshold   │ 2%         │');
    console.log('│ Trailing Stop        │ 8%         │');
    console.log('│ Max Position         │ 30%        │');
    console.log('│ Rebalance Interval   │ 24 hours   │');
    console.log('│ Risk Per Trade       │ 2%         │');
    console.log('└─────────────────────────────────────────────────────┘');
    
    db.close();
}

main();