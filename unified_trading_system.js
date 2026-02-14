// Unified Trading System - Kraken + Coinbase + Backtesting
// Combines Kraken bot with Coinbase integration and backtesting capabilities
// Supports both live trading and simulation mode

// ---- CORE IMPORTS ----
import dotenv from 'dotenv';
import { Buffer } from 'buffer';
import crypto from 'crypto';
import readline from 'readline';
import fs from 'fs';
import path from 'path';

// ---- IMPORTS FOR APIs ----
import WebSocket from 'ws';
const fetch = global.fetch || (await import('node-fetch')).default;
import chalk from 'chalk';

// ---- IMPORT THE LOGGER ----
import { appendTradeHistory } from './tradeHistory.js';

dotenv.config();

// ============== Unified Configuration ==============
const UNIFIED_CONFIG = {
    // Trading Parameters
    INITIAL_BALANCE: 10000,
    QUOTE_CURRENCY: 'USD',
    
    // Fee Structure (Inferred from Coinbase Advanced Trade and Kraken)
    FEES: {
        MAKER: 0.004,     // 0.4% maker fee
        TAKER: 0.006,     // 0.6% taker fee
        SPREAD: 0.0025,   // 0.25% spread
        NETWORK: {
            BTC: 0.0001,
            ETH: 0.00005,
            USDC: 0
        }
    },
    
    // Strategy Parameters (Shared between Kraken and Coinbase)
    STRATEGY: {
        // Individual Asset Harvest
        HARVEST_EXCLUDE: ["BTC", "USDC", "USD"],
        FLAT_HARVEST_TRIGGER_PERCENT: 0.03,
        HARVEST_CYCLE_THRESHOLD: 3,
        MIN_SURPLUS_FOR_HARVEST: 1.00,
        MIN_SURPLUS_FOR_FORCED_HARVEST: 1.00,
        FORCED_HARVEST_TIMEOUT: 20 * 60 * 1000,
        
        // Portfolio Override Harvest
        ENABLE_PORTFOLIO_HARVEST: true,
        PORTFOLIO_HARVEST_TRIGGER_DEVIATION_PERCENT: 0.05,
        PORTFOLIO_HARVEST_CONFIRMATION_CYCLES: 3,
        MIN_ASSET_SURPLUS_FOR_PORTFOLIO_HARVEST: 0.10,
        
        // Harvest Proceeds Allocation
        HARVEST_ALLOC_BTC_PERCENT: 0.10,
        HARVEST_ALLOC_REINVEST_PERCENT: 0.50,
        HARVEST_ALLOC_CASH_PERCENT: 0.40,
        MIN_HARVEST_TO_ALLOCATE: 1.00,
        MIN_NEGATIVE_DEVIATION_FOR_REINVEST: -0.01,
        MIN_REINVEST_BUY_USD: 0.50,
        MIN_BTC_BUY_USD: 9999.10, // Keep BTC effectively disabled
        
        // Rebalance
        REBALANCE_EXCLUDE: ["BTC", "USDC", "USD"],
        FLAT_REBALANCE_TRIGGER_PERCENT: 0.04,
        PARTIAL_RECOVERY_PERCENT: 0.875,
        REBALANCE_POSITIVE_THRESHOLD: 3,
        MAX_REBALANCE_ATTEMPTS: 3,
        REBALANCE_COOLDOWN: 30 * 60 * 1000,
        FORCE_REBALANCE_TIMEOUT: 25 * 60 * 1000,
        FORCE_REBALANCE_SHORTFALL_PERCENT: 0.25,
        MIN_PARTIAL_REBALANCE_USD: 1.00,
        MIN_FORCED_REBALANCE_USD: 1.00,
        
        // Adaptive Dead Zone
        ENABLE_ADAPTIVE_DEAD_ZONE: true,
        ADAPTIVE_DZ_INACTIVITY_TIMEOUT: 3 * 60 * 60 * 1000,
        ADAPTIVE_DZ_HARVEST_TRIGGER_PERCENT: 0.020,
        ADAPTIVE_DZ_REBALANCE_TRIGGER_PERCENT: 0.020,
        
        // Crash Protection
        ENABLE_CRASH_PROTECTION: true,
        CP_TRIGGER_ASSET_PERCENT: 0.70,
        CP_TRIGGER_MIN_NEGATIVE_DEV_PERCENT: -0.01,
        CRASH_PROTECTION_THRESHOLD_INCREASE: 2,
        CRASH_PROTECTION_PARTIAL_RECOVERY_PERCENT: 0.55,
        
        // Core Strategy
        TARGET_ADJUST_PERCENT: 0.000,
        
        // Timing
        REFRESH_INTERVAL: 8000,
        BACKTEST_INTERVAL: 5000
    }
};

// ============== Exchange Maps ==============
const EXCHANGE_MAPS = {
    KRAKEN: {
        ASSETS: {
            'XXBT': 'BTC', 'ETH': 'ETH', 'USDC': 'USDC', 'SOL': 'SOL', 'ADA': 'ADA',
            'XLM': 'XLM', 'AVAX': 'AVAX', 'XRP': 'XRP', 'LINK': 'LINK', 'UNI': 'UNI',
            'DOGE': 'DOGE', 'SHIB': 'SHIB', 'PEPE': 'PEPE', 'BONK': 'BONK', 'WIF': 'WIF',
            'AAVE': 'AAVE', 'COMP': 'COMP', 'POPCAT': 'POPCAT', 'ALGO': 'ALGO', 'FET': 'FET',
            'ICP': 'ICP', 'NEAR': 'NEAR', 'RENDER': 'RENDER', 'TAO': 'TAO', 'XLTC': 'LTC',
            'TRX': 'TRX', 'SUI': 'SUI', 'XXDG': 'XDG', 'ZUSD': 'USD', 'INJ': 'INJ',
            'OCEAN': 'OCEAN', 'TRUMP': 'TRUMP', 'XBT': 'BTC', 'XETH': 'ETH', 'XXRP': 'XRP'
        },
        QUOTE_ASSET: 'ZUSD'
    },
    COINBASE: {
        ASSETS: {
            'BTC': 'BTC', 'ETH': 'ETH', 'USDC': 'USDC', 'SOL': 'SOL', 'ADA': 'ADA',
            'XLM': 'XLM', 'AVAX': 'AVAX', 'XRP': 'XRP', 'LINK': 'LINK', 'UNI': 'UNI',
            'DOGE': 'DOGE', 'SHIB': 'SHIB', 'PEPE': 'PEPE', 'BONK': 'BONK', 'WIF': 'WIF',
            'AAVE': 'AAVE', 'COMP': 'COMP', 'POPCAT': 'POPCAT', 'ALGO': 'ALGO', 'FET': 'FET',
            'ICP': 'ICP', 'NEAR': 'NEAR', 'RENDER': 'RENDER', 'TAO': 'TAO', 'LTC': 'LTC',
            'TRX': 'TRX', 'SUI': 'SUI', 'XDG': 'XDG', 'USD': 'USD', 'INJ': 'INJ',
            'OCEAN': 'OCEAN', 'TRUMP': 'TRUMP'
        },
        QUOTE_ASSET: 'USD'
    }
};

// ============== Base Exchange API Class ==============
class ExchangeAPI {
    constructor(name, config) {
        this.name = name;
        this.config = config;
        this.connected = false;
        this.latestPrices = new Map();
        this.pairInfo = new Map();
        this.assetInfo = new Map();
    }
    
    async initialize() {
        throw new Error('initialize() must be implemented by subclass');
    }
    
    async getHoldings() {
        throw new Error('getHoldings() must be implemented by subclass');
    }
    
    getLatestPrice(symbol) {
        return this.latestPrices.get(symbol);
    }
    
    getPairData(symbol) {
        return this.pairInfo.get(symbol);
    }
    
    roundQty(symbol, quantity) {
        throw new Error('roundQty() must be implemented by subclass');
    }
    
    async placeBuy(symbol, quantity) {
        throw new Error('placeBuy() must be implemented by subclass');
    }
    
    async placeSell(symbol, quantity) {
        throw new Error('placeSell() must be implemented by subclass');
    }
    
    subscribeToTickers(symbols) {
        throw new Error('subscribeToTickers() must be implemented by subclass');
    }
    
    isWsConnected() {
        return this.connected;
    }
    
    close() {
        // Override in subclass if needed
    }
    
    // Helper methods
    getSymbolFromExchange(exchangeSymbol) {
        return exchangeSymbol;
    }
    
    getExchangeSymbol(symbol) {
        return symbol;
    }
}

// ============== Kraken API Implementation ==============
class KrakenAPI extends ExchangeAPI {
    constructor(apiKey, apiSecret) {
        super('Kraken', UNIFIED_CONFIG);
        this.apiKey = apiKey;
        this.apiSecret = apiSecret;
        this.ws = null;
        this.wsConnected = false;
        this.secretBuffer = Buffer.from(apiSecret, 'base64');
        this.lastNonce = Date.now() * 1000;
        this.activeSubscriptions = new Set();
        this.wsToken = null;
        this.wsTokenExpires = 0;
    }
    
    _getNonce() {
        this.lastNonce = Math.max(this.lastNonce + 1, Date.now() * 1000);
        return this.lastNonce;
    }
    
    _signRequest(path, bodyString, nonce) {
        try {
            const hash = crypto.createHash('sha256');
            const hmac = crypto.createHmac('sha512', this.secretBuffer);
            const hashDigest = hash.update(nonce.toString() + bodyString).digest('binary');
            const pathBuffer = Buffer.from(path);
            const hmacInput = Buffer.concat([pathBuffer, Buffer.from(hashDigest, 'binary')]);
            return hmac.update(hmacInput).digest('base64');
        } catch (e) {
            console.error("Error during request signing:", e);
            throw new Error("Failed to sign request.");
        }
    }
    
    async _request(endpoint, params = {}, isPrivate = false, method = 'POST') {
        const path = `/0/${isPrivate ? 'private' : 'public'}/${endpoint}`;
        const url = `https://api.kraken.com${path}`;
        const headers = { 'User-Agent': 'UnifiedTradingSystem/1.0' };
        let bodyString = '';
        let nonce;
        
        if (isPrivate) {
            nonce = this._getNonce();
            params.nonce = nonce;
            bodyString = new URLSearchParams(params).toString();
            headers['API-Key'] = this.apiKey;
            headers['API-Sign'] = this._signRequest(path, bodyString, nonce);
            headers['Content-Type'] = 'application/x-www-form-urlencoded; charset=utf-8';
        }
        
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 20000);
        
        try {
            const response = await fetch(url, {
                method: method,
                headers: headers,
                body: method !== 'GET' && bodyString ? bodyString : undefined,
                signal: controller.signal
            });
            clearTimeout(timeoutId);
            
            const responseBody = await response.text();
            if (!response.ok) {
                console.error(`HTTP Error ${response.status} on ${method} ${url}`);
                throw new Error(`HTTP Error ${response.status}: ${response.statusText}`);
            }
            
            const data = JSON.parse(responseBody);
            if (data.error && data.error.length > 0) {
                const errorMsg = data.error.join(', ');
                throw new Error(`Kraken API Error (${endpoint}): ${errorMsg}`);
            }
            
            return data.result;
        } catch (error) {
            clearTimeout(timeoutId);
            if (error.name === 'AbortError') {
                throw new Error(`Request (${endpoint}) timed out`);
            }
            throw error;
        }
    }
    
    async initialize() {
        console.log("🔌 Initializing Kraken API...");
        try {
            // Fetch asset and pair info
            const [assetResult, pairResult] = await Promise.all([
                this._request('Assets', {}, false, 'GET'),
                this._request('AssetPairs', {}, false, 'GET')
            ]);
            
            // Store asset info
            for (const [krakenSymbol, data] of Object.entries(assetResult)) {
                const commonSymbol = EXCHANGE_MAPS.KRAKEN.ASSETS[krakenSymbol] || data.altname;
                this.assetInfo.set(commonSymbol, { krakenSymbol, ...data });
            }
            
            // Store pair info
            for (const [pairKey, data] of Object.entries(pairResult)) {
                const baseCommon = EXCHANGE_MAPS.KRAKEN.ASSETS[data.base] || data.base;
                const quoteCommon = EXCHANGE_MAPS.KRAKEN.ASSETS[data.quote] || data.quote;
                this.pairInfo.set(baseCommon, { krakenPair: pairKey, ...data });
            }
            
            console.log(`✅ Kraken: ${this.assetInfo.size} assets, ${this.pairInfo.size} pairs`);
            
            // Connect WebSocket
            this._connectWebSocket();
            this.connected = true;
        } catch (error) {
            console.error("❌ Failed to initialize Kraken:", error.message);
            throw error;
        }
    }
    
    _connectWebSocket() {
        console.log("🔌 Connecting to Kraken WebSocket...");
        this.ws = new WebSocket('wss://ws.kraken.com');
        
        this.ws.on('open', () => {
            console.log("✅ Kraken WebSocket connected");
            this.wsConnected = true;
        });
        
        this.ws.on('message', (data) => {
            try {
                const message = JSON.parse(data.toString());
                
                if (message.event === 'subscriptionStatus') {
                    if (message.status === 'subscribed') {
                        console.log(`✅ Subscribed to ${message.pair}`);
                        this.activeSubscriptions.add(message.pair);
                    } else if (message.status === 'error') {
                        console.error(`❌ Subscription error: ${message.errorMessage}`);
                    }
                }
                
                if (Array.isArray(message) && message[2] === 'ticker') {
                    const pair = message[3];
                    const priceStr = message[1]?.c?.[0];
                    const price = parseFloat(priceStr);
                    
                    if (!isNaN(price) && price > 0) {
                        // Find common symbol for this pair
                        for (const [common, info] of this.pairInfo.entries()) {
                            if (info.krakenPair === pair) {
                                this.latestPrices.set(common, price);
                                break;
                            }
                        }
                    }
                }
            } catch (error) {
                console.error("Error processing WebSocket message:", error);
            }
        });
        
        this.ws.on('error', (err) => {
            console.error("Kraken WebSocket error:", err.message);
        });
        
        this.ws.on('close', () => {
            console.log("❌ Kraken WebSocket closed");
            this.wsConnected = false;
            this.activeSubscriptions.clear();
            this.latestPrices.clear();
            
            // Reconnect after delay
            setTimeout(() => this._connectWebSocket(), 5000);
        });
    }
    
    async getHoldings() {
        try {
            const balances = await this._request('Balance', {}, true);
            let quoteBalance = 0;
            const holdingsMap = new Map();
            
            for (const [krakenSymbol, qtyStr] of Object.entries(balances)) {
                const quantity = parseFloat(qtyStr);
                if (isNaN(quantity) || quantity <= 0) continue;
                
                const commonSymbol = EXCHANGE_MAPS.KRAKEN.ASSETS[krakenSymbol];
                if (!commonSymbol) continue;
                
                if (commonSymbol === 'USD') {
                    quoteBalance += quantity;
                } else if (commonSymbol === 'USDC') {
                    quoteBalance += quantity;
                } else if (commonSymbol && quantity > 0.00000001) {
                    if (holdingsMap.has(commonSymbol)) {
                        holdingsMap.get(commonSymbol).quantity += quantity;
                    } else {
                        holdingsMap.set(commonSymbol, { symbol: commonSymbol, quantity });
                    }
                }
            }
            
            const holdingsArray = Array.from(holdingsMap.values());
            return [quoteBalance, holdingsArray];
        } catch (error) {
            console.error("Error fetching Kraken holdings:", error.message);
            throw error;
        }
    }
    
    async fetchOHLC(symbol, since = null, interval = 60) {
        const pairInfo = this.pairInfo.get(symbol);
        if (!pairInfo) {
            throw new Error(`No pair info for ${symbol}`);
        }
        
        const params = { pair: pairInfo.krakenPair, interval };
        if (since) {
            params.since = Math.floor(since / 1000);
        }
        
        const result = await this._request('OHLC', params, false, 'GET');
        const pairKey = Object.keys(result)[0];
        const ohlcData = result[pairKey];
        
        return ohlcData.map(candle => ({
            time: candle[0] * 1000,
            open: parseFloat(candle[1]),
            high: parseFloat(candle[2]),
            low: parseFloat(candle[3]),
            close: parseFloat(candle[4]),
            volume: parseFloat(candle[5])
        }));
    }
    
    subscribeToTickers(symbols) {
        if (!this.wsConnected || !this.ws || this.ws.readyState !== WebSocket.OPEN) {
            console.warn("Kraken WebSocket not ready for subscription");
            return;
        }
        
        const pairsToSubscribe = [];
        for (const symbol of symbols) {
            const pairInfo = this.pairInfo.get(symbol);
            if (pairInfo && pairInfo.krakenPair && !this.activeSubscriptions.has(pairInfo.krakenPair)) {
                pairsToSubscribe.push(pairInfo.krakenPair);
            }
        }
        
        if (pairsToSubscribe.length > 0) {
            const message = {
                event: 'subscribe',
                pair: pairsToSubscribe,
                subscription: { name: 'ticker' }
            };
            this.ws.send(JSON.stringify(message));
        }
    }
    
    roundQty(symbol, quantity) {
        const pairInfo = this.pairInfo.get(symbol);
        if (!pairInfo || !pairInfo.lot_decimals) {
            // Fallback: round to 8 decimals for crypto
            return parseFloat(quantity).toFixed(8);
        }
        
        const decimals = pairInfo.lot_decimals;
        const factor = Math.pow(10, decimals);
        const rounded = Math.floor(quantity * factor) / factor;
        return rounded.toFixed(decimals);
    }
    
    async placeBuy(symbol, quantity) {
        const pairInfo = this.pairInfo.get(symbol);
        if (!pairInfo) {
            throw new Error(`No Kraken pair found for ${symbol}`);
        }
        
        try {
            const response = await this._request('AddOrder', {
                pair: pairInfo.krakenPair,
                type: 'buy',
                ordertype: 'market',
                volume: quantity.toString()
            }, true);
            
            const orderId = response.txid?.[0] || `kraken_${crypto.randomUUID()}`;
            return { id: orderId, description: response.descr?.order || 'Kraken Buy' };
        } catch (error) {
            console.error(`❌ Kraken Buy failed for ${symbol}:`, error.message);
            throw error;
        }
    }
    
    async placeSell(symbol, quantity) {
        const pairInfo = this.pairInfo.get(symbol);
        if (!pairInfo) {
            throw new Error(`No Kraken pair found for ${symbol}`);
        }
        
        try {
            const response = await this._request('AddOrder', {
                pair: pairInfo.krakenPair,
                type: 'sell',
                ordertype: 'market',
                volume: quantity.toString()
            }, true);
            
            const orderId = response.txid?.[0] || `kraken_${crypto.randomUUID()}`;
            return { id: orderId, description: response.descr?.order || 'Kraken Sell' };
        } catch (error) {
            console.error(`❌ Kraken Sell failed for ${symbol}:`, error.message);
            throw error;
        }
    }
    
    close() {
        if (this.ws) {
            this.ws.close();
        }
        this.connected = false;
    }
}

// ============== Coinbase API Implementation ==============
class CoinbaseAPI extends ExchangeAPI {
    constructor(apiKey, apiSecret, passphrase = '') {
        super('Coinbase', UNIFIED_CONFIG);
        this.apiKey = apiKey;
        this.apiSecret = apiSecret;
        this.passphrase = passphrase;
        this.ws = null;
        this.wsConnected = false;
        this.hmacSecret = null;
        if (apiSecret) {
            this.hmacSecret = this._extractHmacSecret(apiSecret);
            if (!this.hmacSecret && typeof apiSecret === 'string') {
                this.hmacSecret = Buffer.from(apiSecret, 'utf-8');
            }
        }
        this.productPrices = new Map();
    }
    
    _extractHmacSecret(apiSecret) {
        if (!apiSecret || apiSecret.includes('BEGIN EC PRIVATE KEY')) {
            // Extract 32-byte private key from EC private key
            try {
                const ecKeyStr = apiSecret.replace('\\n', '\n');
                const lines = ecKeyStr.split('\n');
                const keyLines = [];
                
                let inKey = false;
                for (const line of lines) {
                    const trimmed = line.trim();
                    if (trimmed.includes('BEGIN EC PRIVATE KEY')) {
                        inKey = true;
                        continue;
                    } else if (trimmed.includes('END EC PRIVATE KEY')) {
                        break;
                    } else if (inKey && trimmed) {
                        keyLines.push(trimmed);
                    }
                }
                
                const keyDataB64 = keyLines.join('');
                const keyBytes = Buffer.from(keyDataB64, 'base64');
                
                // Find 32-byte private key in ASN.1 structure
                for (let i = 0; i < keyBytes.length - 33; i++) {
                    if (keyBytes[i] === 0x04 && keyBytes[i + 1] === 32) {
                        return keyBytes.slice(i + 2, i + 34);
                    }
                }
            } catch (error) {
                console.error("Error extracting HMAC secret:", error);
            }
        }
        return null;
    }
    
    async initialize() {
        console.log("🔌 Initializing Coinbase API...");
        try {
            // Fetch products
            const response = await fetch('https://api.coinbase.com/v2/exchange-rates');
            const data = await response.json();
            
            if (data.data && data.data.rates) {
                for (const [symbol, rate] of Object.entries(data.data.rates)) {
                    const commonSymbol = EXCHANGE_MAPS.COINBASE.ASSETS[symbol];
                    if (commonSymbol) {
                        this.pairInfo.set(commonSymbol, { symbol: commonSymbol, price: parseFloat(rate) });
                    }
                }
            }
            
            console.log(`✅ Coinbase: ${this.pairInfo.size} pairs loaded`);
            
            // Connect WebSocket for real-time prices
            this._connectWebSocket();
            this.connected = true;
        } catch (error) {
            console.error("❌ Failed to initialize Coinbase:", error.message);
            throw error;
        }
    }
    
    _connectWebSocket() {
        console.log("🔌 Connecting to Coinbase WebSocket...");
        this.ws = new WebSocket('wss://ws-feed.exchange.coinbase.com');
        
        this.ws.on('open', () => {
            console.log("✅ Coinbase WebSocket connected");
            this.wsConnected = true;
            
            // Subscribe to ticker channels
            const products = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD'];
            const subscribeMsg = {
                type: 'subscribe',
                product_ids: products,
                channels: ['ticker']
            };
            this.ws.send(JSON.stringify(subscribeMsg));
        });
        
        this.ws.on('message', (data) => {
            try {
                const message = JSON.parse(data.toString());
                
                if (message.type === 'ticker') {
                    const price = parseFloat(message.price);
                    const product = message.product_id;
                    
                    if (!isNaN(price) && price > 0) {
                        // Extract base symbol from product ID
                        const baseSymbol = product.split('-')[0];
                        const commonSymbol = EXCHANGE_MAPS.COINBASE.ASSETS[baseSymbol];
                        if (commonSymbol) {
                            this.latestPrices.set(commonSymbol, price);
                        }
                    }
                }
            } catch (error) {
                console.error("Error processing Coinbase WebSocket message:", error);
            }
        });
        
        this.ws.on('error', (err) => {
            console.error("Coinbase WebSocket error:", err.message);
        });
        
        this.ws.on('close', () => {
            console.log("❌ Coinbase WebSocket closed");
            this.wsConnected = false;
            this.latestPrices.clear();
            
            // Reconnect after delay
            setTimeout(() => this._connectWebSocket(), 5000);
        });
    }
    
    async getHoldings() {
        // For Coinbase, we'll use a simulation approach since real API requires different auth
        // This would be replaced with actual Coinbase API calls in production
        const simulationMode = !this.apiKey || !this.apiSecret;
        
        if (simulationMode) {
            // Return simulated holdings (for backtesting/demo)
            return [5000, [
                { symbol: 'BTC', quantity: 0.01 },
                { symbol: 'ETH', quantity: 0.5 },
                { symbol: 'SOL', quantity: 2 }
            ]];
        }
        
        // Real API call would go here
        try {
            // This is a placeholder - would need proper Coinbase API authentication
            const response = await fetch('https://api.coinbase.com/v2/accounts', {
                headers: {
                    'Authorization': `Bearer ${this.apiKey}`,
                    'CB-ACCESS-SIGN': this._generateSignature(),
                    'CB-ACCESS-TIMESTAMP': Date.now().toString(),
                    'CB-ACCESS-KEY': this.apiKey,
                    'CB-ACCESS-PASSPHRASE': this.passphrase
                }
            });
            
            const data = await response.json();
            // Process data and return holdings
            return [0, []]; // Placeholder
        } catch (error) {
            console.error("Error fetching Coinbase holdings:", error);
            return [0, []];
        }
    }
    
    _generateSignature() {
        return 'signature_placeholder';
    }
    
    async fetchCandles(symbol, granularity = 3600, start = null, end = null) {
        const productId = `${symbol}-USD`;
        const now = Math.floor(Date.now() / 1000);
        const startTs = start || (now - 30 * 24 * 60 * 60);
        const endTs = end || now;
        
        const maxCandles = 300;
        const chunkSize = maxCandles * granularity;
        
        let allCandles = [];
        let chunkStart = startTs;
        
        while (chunkStart < endTs) {
            const chunkEnd = Math.min(chunkStart + chunkSize, endTs);
            const url = `https://api.exchange.coinbase.com/products/${productId}/candles?granularity=${granularity}&start=${chunkStart}&end=${chunkEnd}`;
            
            console.log(`  Fetching chunk: ${new Date(chunkStart * 1000).toISOString()} to ${new Date(chunkEnd * 1000).toISOString()}`);
            const response = await fetch(url);
            
            if (!response.ok) {
                const text = await response.text();
                throw new Error(`Coinbase candles error: ${response.status} - ${text}`);
            }
            
            const data = await response.json();
            if (Array.isArray(data) && data.length > 0) {
                allCandles = allCandles.concat(data);
            }
            
            chunkStart = chunkEnd;
            await new Promise(r => setTimeout(r, 250));
        }
        
        return allCandles.map(candle => ({
            time: candle[0] * 1000,
            low: parseFloat(candle[1]),
            high: parseFloat(candle[2]),
            open: parseFloat(candle[3]),
            close: parseFloat(candle[4]),
            volume: parseFloat(candle[5])
        })).sort((a, b) => a.time - b.time);
    }
    
    subscribeToTickers(symbols) {
        if (!this.wsConnected || !this.ws || this.ws.readyState !== WebSocket.OPEN) {
            console.warn("Coinbase WebSocket not ready for subscription");
            return;
        }
        
        const products = [];
        for (const symbol of symbols) {
            products.push(`${symbol}-USD`);
        }
        
        if (products.length > 0) {
            const subscribeMsg = {
                type: 'subscribe',
                product_ids: products,
                channels: ['ticker']
            };
            this.ws.send(JSON.stringify(subscribeMsg));
        }
    }
    
    roundQty(symbol, quantity) {
        // Coinbase typically uses 8 decimal places for crypto
        const decimals = 8;
        const factor = Math.pow(10, decimals);
        const rounded = Math.floor(quantity * factor) / factor;
        return rounded.toFixed(decimals);
    }
    
    async placeBuy(symbol, quantity) {
        // For simulation/demo mode
        const simulationMode = !this.apiKey || !this.apiSecret;
        
        if (simulationMode) {
            return { id: `sim_${crypto.randomUUID()}`, description: `Simulated Coinbase Buy ${quantity} ${symbol}` };
        }
        
        // Real API call would go here
        throw new Error("Coinbase live trading not implemented - use simulation mode");
    }
    
    async placeSell(symbol, quantity) {
        // For simulation/demo mode
        const simulationMode = !this.apiKey || !this.apiSecret;
        
        if (simulationMode) {
            return { id: `sim_${crypto.randomUUID()}`, description: `Simulated Coinbase Sell ${quantity} ${symbol}` };
        }
        
        // Real API call would go here
        throw new Error("Coinbase live trading not implemented - use simulation mode");
    }
    
    close() {
        if (this.ws) {
            this.ws.close();
        }
        this.connected = false;
    }
}

// ============== Backtesting Engine ==============
class BacktestingEngine {
    constructor(strategyConfig) {
        this.config = strategyConfig;
        this.historicalData = new Map(); // symbol -> [{time, price}]
        this.simulatedHoldings = new Map();
        this.simulatedBalance = UNIFIED_CONFIG.INITIAL_BALANCE;
        this.simulatedState = {
            baselines: {},
            trailingState: {},
            lastActionTimestamps: {},
            rebalanceState: {},
            adaptiveDeadZoneState: {},
            portfolioHarvestState: { flagged: false, cycleCount: 0 }
        };
        this.tradeLog = [];
        this.currentTime = Date.now();
    }
    
    seedInitialHoldings(holdingsMap) {
        for (const [symbol, quantity] of Object.entries(holdingsMap)) {
            this.simulatedHoldings.set(symbol, quantity);
            const price = this._getInitialPrice(symbol);
            const value = quantity * price;
            this.simulatedState.baselines[symbol] = value;
            this.simulatedBalance -= value;
            console.log(`🌱 Seeded ${quantity} ${symbol} @ $${price} = $${value.toFixed(2)}`);
        }
    }
    
    loadHistoricalData(symbol, dataPoints) {
        this.historicalData.set(symbol, dataPoints.sort((a, b) => a.time - b.time));
    }
    
    async fetchHistoricalData(exchangeApi, symbol, days = 90, exchange = 'coinbase') {
        console.log(`📥 Fetching ${days} days of historical data for ${symbol} from ${exchange}...`);
        
        const endTime = Math.floor(Date.now() / 1000);
        const startTime = endTime - (days * 24 * 60 * 60);
        
        let candles = [];
        
        if (exchange === 'coinbase' && exchangeApi.fetchCandles) {
            candles = await exchangeApi.fetchCandles(symbol, 3600, startTime, endTime);
        } else if (exchange === 'kraken' && exchangeApi.fetchOHLC) {
            candles = await exchangeApi.fetchOHLC(symbol, startTime * 1000, 60);
        } else {
            throw new Error(`Unsupported exchange: ${exchange}`);
        }
        
        const dataPoints = candles.map(c => ({
            time: c.time,
            price: c.close
        }));
        
        this.historicalData.set(symbol, dataPoints);
        console.log(`📊 Loaded ${dataPoints.length} data points for ${symbol}`);
        return dataPoints.length;
    }
    
    generateSyntheticData(symbol, days = 30, volatility = 0.02) {
        const dataPoints = [];
        const now = Date.now();
        const initialPrice = this._getInitialPrice(symbol);
        
        let currentPrice = initialPrice;
        for (let i = days * 24 * 60; i >= 0; i--) {
            const time = now - (i * 60 * 1000); // 1-minute intervals
            const change = (Math.random() - 0.5) * volatility * currentPrice;
            currentPrice += change;
            currentPrice = Math.max(currentPrice, initialPrice * 0.5); // Don't let it drop too much
            currentPrice = Math.min(currentPrice, initialPrice * 2); // Don't let it go too high
            
            dataPoints.push({ time, price: currentPrice });
        }
        
        this.historicalData.set(symbol, dataPoints);
        console.log(`📊 Generated ${dataPoints.length} data points for ${symbol}`);
    }
    
    _getInitialPrice(symbol) {
        const prices = {
            'BTC': 68000,
            'ETH': 2000,
            'SOL': 80,
            'ADA': 0.25,
            'XRP': 1.40,
            'USDC': 1.0,
            'USD': 1.0
        };
        return prices[symbol] || 100;
    }
    
    _getCurrentPrice(symbol, currentTime = this.currentTime) {
        const data = this.historicalData.get(symbol);
        if (!data || data.length === 0) return null;
        
        // Find the closest data point before currentTime
        let closest = data[0];
        for (const point of data) {
            if (point.time <= currentTime) {
                closest = point;
            } else {
                break;
            }
        }
        
        return closest.price;
    }
    
    runBacktest(startTime, endTime, interval = 60000) {
        console.log(`\n🚀 Starting backtest from ${new Date(startTime).toISOString()} to ${new Date(endTime).toISOString()}`);
        
        let currentTime = startTime;
        let cycleCount = 0;
        
        while (currentTime <= endTime && cycleCount < 1000) { // Limit cycles for demo
            cycleCount++;
            console.log(`\n----- Backtest Cycle ${cycleCount} at ${new Date(currentTime).toISOString()} -----`);
            
            // Get current prices for all symbols
            const currentPrices = new Map();
            for (const [symbol, data] of this.historicalData.entries()) {
                const price = this._getCurrentPrice(symbol, currentTime);
                if (price) {
                    currentPrices.set(symbol, price);
                }
            }
            
            // Update simulated holdings value
            this._updatePortfolioValue(currentPrices);
            
            // Run trading logic
            this._runTradingLogic(currentPrices, currentTime);
            
            // Advance time
            currentTime += interval;
            this.currentTime = currentTime;
        }
        
        console.log(`\n✅ Backtest completed: ${cycleCount} cycles`);
        this._generateBacktestReport();
    }
    
    _updatePortfolioValue(currentPrices) {
        let totalValue = this.simulatedBalance;
        
        for (const [symbol, quantity] of this.simulatedHoldings.entries()) {
            const price = currentPrices.get(symbol);
            if (price) {
                totalValue += quantity * price;
            }
        }
        
        return totalValue;
    }
    
    _runTradingLogic(currentPrices, currentTime) {
        const config = this.config;
        const s = this.simulatedState;
        
        // Convert simulated holdings to portfolio summary format
        const portfolioSummary = [];
        let totalHoldingsValue = 0;
        
        for (const [symbol, quantity] of this.simulatedHoldings.entries()) {
            const price = currentPrices.get(symbol);
            if (!price) continue;
            
            const value = quantity * price;
            totalHoldingsValue += value;
            
            const baseline = s.baselines[symbol];
            let deviation = null;
            if (baseline && baseline > 0) {
                deviation = (value - baseline) / baseline;
            }
            
            portfolioSummary.push({
                Symbol: symbol,
                Quantity: quantity,
                Price: price,
                Value: value,
                Baseline: baseline,
                Deviation: deviation,
                rawQuantity: quantity,
                currentPrice: price,
                usdValueNum: value
            });
        }
        
        const totalPortfolioValue = this.simulatedBalance + totalHoldingsValue;
        const managedAssets = portfolioSummary.filter(row => !config.STRATEGY.REBALANCE_EXCLUDE.includes(row.Symbol));
        let totalBaselineDifference = 0;
        let totalBaselineValue = 0;
        
        for (const asset of managedAssets) {
            if (asset.Baseline && asset.Baseline > 0) {
                totalBaselineDifference += (asset.Value - asset.Baseline);
                totalBaselineValue += asset.Baseline;
            }
        }
        
        const portfolioDeviationPercent = totalBaselineValue > 0 ? (totalBaselineDifference / totalBaselineValue) * 100 : 0;
        
        console.log(`📊 Portfolio: $${totalPortfolioValue.toFixed(2)} (Balance: $${this.simulatedBalance.toFixed(2)}, Holdings: $${totalHoldingsValue.toFixed(2)})`);
        console.log(`📈 Portfolio Deviation: ${portfolioDeviationPercent.toFixed(2)}%`);
        
        // Run trading logic (simplified version)
        this._runTradingLogicSimplified(portfolioSummary, currentTime);
    }
    
    _runTradingLogicSimplified(portfolioSummary, currentTime) {
        const config = this.config.STRATEGY;
        
        // Initialize baselines
        for (const row of portfolioSummary) {
            const symbol = row.Symbol;
            if (!this.simulatedState.baselines[symbol] && row.Value > 0.01) {
                this.simulatedState.baselines[symbol] = row.Value;
                console.log(`✨ Initialized baseline for ${symbol}: $${row.Value.toFixed(2)}`);
            }
        }
        
        // Check for harvest opportunities
        for (const row of portfolioSummary) {
            const symbol = row.Symbol;
            if (config.HARVEST_EXCLUDE.includes(symbol)) continue;
            
            const baseline = this.simulatedState.baselines[symbol];
            if (!baseline || baseline <= 0) continue;
            
            const deviation = row.Deviation;
            if (deviation === null || deviation === undefined) continue;
            
            // Check if asset is ready for harvest
            if (deviation >= config.FLAT_HARVEST_TRIGGER_PERCENT) {
                const surplus = row.Value - baseline;
                if (surplus >= config.MIN_SURPLUS_FOR_HARVEST) {
                    console.log(`💰 Harvest opportunity for ${symbol}: $${surplus.toFixed(2)} surplus`);
                    
                    // Simulate sell
                    const sellQuantity = surplus / row.Price;
                    this._executeSimulatedTrade('SELL', symbol, sellQuantity, row.Price);
                    
                    // Update baseline
                    this.simulatedState.baselines[symbol] = baseline * (1 + config.TARGET_ADJUST_PERCENT);
                }
            }
        }
    }
    
    _executeSimulatedTrade(side, symbol, quantity, price) {
        const value = quantity * price;
        const feeRate = side === 'BUY' ? this.config.FEES.TAKER : this.config.FEES.MAKER;
        const fee = value * feeRate;
        const totalCost = value + fee;
        
        console.log(`📝 Simulated ${side}: ${quantity.toFixed(8)} ${symbol} @ $${price.toFixed(2)} (Value: $${value.toFixed(2)}, Fee: $${fee.toFixed(2)})`);
        
        if (side === 'BUY') {
            if (this.simulatedBalance < totalCost) {
                console.log(`❌ Insufficient balance for buy: $${totalCost.toFixed(2)} > $${this.simulatedBalance.toFixed(2)}`);
                return false;
            }
            
            this.simulatedBalance -= totalCost;
            const currentQuantity = this.simulatedHoldings.get(symbol) || 0;
            this.simulatedHoldings.set(symbol, currentQuantity + quantity);
            
            // Update baseline
            if (!this.simulatedState.baselines[symbol]) {
                this.simulatedState.baselines[symbol] = totalCost;
            } else {
                const currentTotal = this.simulatedState.baselines[symbol] + totalCost;
                this.simulatedState.baselines[symbol] = currentTotal;
            }
            
        } else if (side === 'SELL') {
            const currentQuantity = this.simulatedHoldings.get(symbol) || 0;
            if (currentQuantity < quantity) {
                console.log(`❌ Insufficient quantity for sell: ${quantity.toFixed(8)} > ${currentQuantity.toFixed(8)}`);
                return false;
            }
            
            this.simulatedHoldings.set(symbol, currentQuantity - quantity);
            this.simulatedBalance += value - fee;
        }
        
        this.tradeLog.push({
            time: this.currentTime,
            side,
            symbol,
            quantity,
            price,
            value,
            fee,
            totalCost
        });
        
        return true;
    }
    
    _generateBacktestReport() {
        console.log(`\n📊 BACKTEST REPORT`);
        console.log(`==================`);
        
        let totalHoldingsValue = 0;
        for (const [symbol, quantity] of this.simulatedHoldings.entries()) {
            if (quantity > 0) {
                const price = this._getCurrentPrice(symbol);
                if (price) {
                    totalHoldingsValue += quantity * price;
                }
            }
        }
        
        const totalPortfolioValue = this.simulatedBalance + totalHoldingsValue;
        const profit = totalPortfolioValue - UNIFIED_CONFIG.INITIAL_BALANCE;
        const profitPercent = (profit / UNIFIED_CONFIG.INITIAL_BALANCE) * 100;
        
        console.log(`Initial Portfolio: $${UNIFIED_CONFIG.INITIAL_BALANCE.toFixed(2)}`);
        console.log(`Final Cash: $${this.simulatedBalance.toFixed(2)}`);
        console.log(`Final Holdings: $${totalHoldingsValue.toFixed(2)}`);
        console.log(`Total Portfolio: $${totalPortfolioValue.toFixed(2)}`);
        console.log(`Profit/Loss: $${profit.toFixed(2)} (${profitPercent.toFixed(2)}%)`);
        
        console.log(`\n📈 Final Holdings:`);
        for (const [symbol, quantity] of this.simulatedHoldings.entries()) {
            if (quantity > 0) {
                const price = this._getCurrentPrice(symbol);
                if (price) {
                    const value = quantity * price;
                    console.log(`  ${symbol}: ${quantity.toFixed(8)} = $${value.toFixed(2)}`);
                }
            }
        }
        
        console.log(`\n📋 Trade History (${this.tradeLog.length} trades):`);
        this.tradeLog.slice(-5).forEach(trade => {
            console.log(`  ${trade.side} ${trade.symbol}: ${trade.quantity.toFixed(8)} @ $${trade.price.toFixed(2)} ($${trade.value.toFixed(2)})`);
        });
        
        console.log(`\n🎯 Strategy Configuration Used:`);
        console.log(`  Harvest Trigger: ${this.config.STRATEGY.FLAT_HARVEST_TRIGGER_PERCENT * 100}%`);
        console.log(`  Rebalance Trigger: ${this.config.STRATEGY.FLAT_REBALANCE_TRIGGER_PERCENT * 100}%`);
        console.log(`  Maker Fee: ${this.config.FEES.MAKER * 100}%`);
        console.log(`  Taker Fee: ${this.config.FEES.TAKER * 100}%`);
    }
}

// ============== Main Unified Trading System ==============
class UnifiedTradingSystem {
    constructor(mode = 'simulation', exchange = 'kraken') {
        this.mode = mode; // 'simulation', 'backtest', 'live'
        this.exchange = exchange; // 'kraken', 'coinbase', 'both'
        this.api = null;
        this.backtestingEngine = null;
        this.running = false;
        
        // Trading state (shared between exchanges)
        this.state = {
            baselines: {},
            trailingState: {},
            lastActionTimestamps: {},
            rebalanceState: {},
            adaptiveDeadZoneState: {},
            portfolioHarvestState: { flagged: false, cycleCount: 0 }
        };
        
        // Holdings (shared)
        this.holdings = new Map();
        this.balance = UNIFIED_CONFIG.INITIAL_BALANCE;
        
        this._loadState();
    }
    
    async initialize() {
        console.log(`🚀 Initializing Unified Trading System (${this.mode} mode, ${this.exchange} exchange)`);
        
        if (this.mode === 'backtest') {
            console.log("🔧 Initializing backtesting engine...");
            this.backtestingEngine = new BacktestingEngine(UNIFIED_CONFIG);
            
            const symbols = ['BTC', 'ETH', 'SOL', 'XRP'];
            const useRealData = process.env.BACKTEST_REAL_DATA === 'true';
            
            if (useRealData) {
                let exchangeApi = null;
                let exchangeName = this.exchange;
                
                if (this.exchange === 'coinbase' || this.exchange === 'simulation') {
                    exchangeApi = new CoinbaseAPI('', '', '');
                    await exchangeApi.initialize().catch(() => {
                        console.log("⚠️  Coinbase API failed, falling back to synthetic data");
                    });
                    exchangeName = 'coinbase';
                } else if (this.exchange === 'kraken') {
                    exchangeApi = new KrakenAPI('', '');
                    await exchangeApi.initialize().catch(() => {
                        console.log("⚠️  Kraken API failed, falling back to synthetic data");
                    });
                    exchangeName = 'kraken';
                }
                
                if (exchangeApi && exchangeApi.connected) {
                    console.log(`📊 Fetching real historical data from ${exchangeName}...`);
                    for (const symbol of symbols) {
                        try {
                            await this.backtestingEngine.fetchHistoricalData(exchangeApi, symbol, 30, exchangeName);
                        } catch (err) {
                            console.log(`⚠️  Failed to fetch ${symbol}, using synthetic: ${err.message}`);
                            this.backtestingEngine.generateSyntheticData(symbol, 30, 0.02);
                        }
                    }
                } else {
                    console.log("📊 Using synthetic data (no API connection)...");
                    symbols.forEach(symbol => {
                        this.backtestingEngine.generateSyntheticData(symbol, 30, 0.02);
                    });
                }
            } else {
                console.log("📊 Using synthetic data (set BACKTEST_REAL_DATA=true for real data)...");
                symbols.forEach(symbol => {
                    this.backtestingEngine.generateSyntheticData(symbol, 30, 0.02);
                });
            }
            
            this.backtestingEngine.seedInitialHoldings({
                'BTC': 0.05,
                'ETH': 1.5,
                'SOL': 25,
                'XRP': 500,
                'USDC': 1000
            });
            
            return;
        }
        
        if (this.exchange === 'kraken' || this.exchange === 'both') {
            const krakenApiKey = process.env.KRAKEN_API_KEY;
            const krakenApiSecret = process.env.KRAKEN_PRIVATE_KEY;
            
            if (krakenApiKey && krakenApiSecret) {
                console.log("🔑 Initializing Kraken API...");
                this.api = new KrakenAPI(krakenApiKey, krakenApiSecret);
                await this.api.initialize();
            } else {
                console.log("⚠️  No Kraken API credentials - using simulation mode");
                this.api = new KrakenAPI('', '');
                await this.api.initialize().catch(() => {});
            }
        }
        
        if (this.exchange === 'coinbase' || this.exchange === 'both') {
            const coinbaseApiKey = process.env.COINBASE_API_KEY;
            const coinbaseApiSecret = process.env.COINBASE_API_SECRET;
            const coinbasePassphrase = process.env.COINBASE_PASSPHRASE || '';
            
            console.log("🔑 Initializing Coinbase API...");
            this.coinbaseApi = new CoinbaseAPI(coinbaseApiKey, coinbaseApiSecret, coinbasePassphrase);
            await this.coinbaseApi.initialize().catch(() => {
                console.log("⚠️  Coinbase API initialization failed - continuing with other exchanges");
            });
        }
    }
    
    async run() {
        this.running = true;
        
        if (this.mode === 'backtest') {
            const startTime = Date.now() - (30 * 24 * 60 * 60 * 1000); // 30 days ago
            const endTime = Date.now();
            this.backtestingEngine.runBacktest(startTime, endTime, 60 * 60 * 1000); // Hourly intervals
            this.running = false;
            return;
        }
        
        console.log("\n🚀 Starting main trading loop...");
        
        while (this.running) {
            const startTime = Date.now();
            console.log(`\n----- Cycle Start: ${new Date().toISOString()} -----`);
            
            try {
                // Fetch holdings and prices
                await this._fetchMarketData();
                
                // Run trading logic
                await this._runTradingLogic();
                
                // Display status
                this._displayStatus();
                
            } catch (error) {
                console.error("❌ Error in main loop:", error.message);
            }
            
            const elapsed = Date.now() - startTime;
            const delay = Math.max(0, UNIFIED_CONFIG.STRATEGY.REFRESH_INTERVAL - elapsed);
            
            console.log(`----- Cycle End: ${elapsed}ms. Waiting ${delay}ms... -----`);
            await this._wait(delay);
        }
        
        console.log("🛑 Trading stopped");
        if (this.api) this.api.close();
        if (this.coinbaseApi) this.coinbaseApi.close();
    }
    
    async _fetchMarketData() {
        // Fetch from all active exchanges
        const exchanges = [];
        if (this.api) exchanges.push(this.api);
        if (this.coinbaseApi) exchanges.push(this.coinbaseApi);
        
        for (const exchange of exchanges) {
            try {
                // Get holdings
                const [cashBalance, holdingsArray] = await exchange.getHoldings();
                
                // Update our combined holdings
                holdingsArray.forEach(holding => {
                    this.holdings.set(holding.symbol, holding.quantity);
                });
                this.balance += cashBalance; // Add to existing balance
                
                // Subscribe to tickers for needed symbols
                const symbols = Array.from(this.holdings.keys());
                if (symbols.length > 0) {
                    exchange.subscribeToTickers(symbols);
                }
                
                // Wait a bit for WebSocket prices
                await this._wait(1000);
                
            } catch (error) {
                console.warn(`⚠️  Error fetching data from ${exchange.name}:`, error.message);
            }
        }
    }
    
    async _runTradingLogic() {
        // Build portfolio summary
        const portfolioSummary = [];
        let totalHoldingsValue = 0;
        
        for (const [symbol, quantity] of this.holdings.entries()) {
            // Get price from first available exchange
            let price = null;
            if (this.api) price = this.api.getLatestPrice(symbol);
            if (!price && this.coinbaseApi) price = this.coinbaseApi.getLatestPrice(symbol);
            
            if (price) {
                const value = quantity * price;
                totalHoldingsValue += value;
                
                const baseline = this.state.baselines[symbol];
                let deviation = null;
                if (baseline && baseline > 0) {
                    deviation = (value - baseline) / baseline;
                }
                
                portfolioSummary.push({
                    Symbol: symbol,
                    Quantity: quantity,
                    Price: price,
                    Value: value,
                    Baseline: baseline,
                    Deviation: deviation,
                    rawQuantity: quantity,
                    currentPrice: price,
                    usdValueNum: value
                });
            }
        }
        
        // Skip if no valid data
        if (portfolioSummary.length === 0) {
            console.log("⚠️  No valid price data available");
            return;
        }
        
        // Run trading strategy (simplified version for demo)
        await this._runTradingStrategy(portfolioSummary);
    }
    
    async _runTradingStrategy(portfolioSummary) {
        const config = UNIFIED_CONFIG.STRATEGY;
        const totalPortfolioValue = this.balance + portfolioSummary.reduce((sum, row) => sum + row.Value, 0);
        
        console.log(`📊 Portfolio Value: $${totalPortfolioValue.toFixed(2)} (Balance: $${this.balance.toFixed(2)})`);
        
        // Initialize baselines
        for (const row of portfolioSummary) {
            const symbol = row.Symbol;
            if (!this.state.baselines[symbol] && row.Value > 0.01) {
                this.state.baselines[symbol] = row.Value;
                console.log(`✨ Initialized baseline for ${symbol}: $${row.Value.toFixed(2)}`);
            }
        }
        
        // Check for harvest opportunities
        for (const row of portfolioSummary) {
            const symbol = row.Symbol;
            if (config.HARVEST_EXCLUDE.includes(symbol)) continue;
            
            const baseline = this.state.baselines[symbol];
            if (!baseline || baseline <= 0) continue;
            
            const deviation = row.Deviation;
            if (deviation === null || deviation === undefined) continue;
            
            // Check if asset is ready for harvest
            if (deviation >= config.FLAT_HARVEST_TRIGGER_PERCENT) {
                const surplus = row.Value - baseline;
                if (surplus >= config.MIN_SURPLUS_FOR_HARVEST) {
                    console.log(`💰 Harvest opportunity for ${symbol}: $${surplus.toFixed(2)} surplus`);
                    
                    // Calculate sell quantity
                    const sellQuantity = surplus / row.Price;
                    const roundedQty = this._roundQuantity(symbol, sellQuantity);
                    
                    if (roundedQty > 0) {
                        console.log(`📈 Selling ${roundedQty} ${symbol} (~$${surplus.toFixed(2)})`);
                        
                        // Execute sell on first available exchange
                        let success = false;
                        if (this.api && this.api.isWsConnected()) {
                            try {
                                const result = await this.api.placeSell(symbol, roundedQty);
                                console.log(`✅ Kraken sell executed: ${result.id}`);
                                success = true;
                            } catch (error) {
                                console.error("❌ Kraken sell failed:", error.message);
                            }
                        }
                        
                        if (!success && this.coinbaseApi) {
                            try {
                                const result = await this.coinbaseApi.placeSell(symbol, roundedQty);
                                console.log(`✅ Coinbase sell executed: ${result.id}`);
                                success = true;
                            } catch (error) {
                                console.error("❌ Coinbase sell failed:", error.message);
                            }
                        }
                        
                        if (success) {
                            // Update baseline
                            this.state.baselines[symbol] = baseline * (1 + config.TARGET_ADJUST_PERCENT);
                            this._saveState();
                        }
                    }
                }
            }
        }
        
        // Check for rebalance opportunities
        for (const row of portfolioSummary) {
            const symbol = row.Symbol;
            if (config.REBALANCE_EXCLUDE.includes(symbol)) continue;
            
            const baseline = this.state.baselines[symbol];
            if (!baseline || baseline <= 0) continue;
            
            const deviation = row.Deviation;
            if (deviation === null || deviation === undefined) continue;
            
            // Check if asset is ready for rebalance
            if (deviation <= -config.FLAT_REBALANCE_TRIGGER_PERCENT) {
                const shortfall = baseline - row.Value;
                if (shortfall >= config.MIN_PARTIAL_REBALANCE_USD && this.balance >= shortfall) {
                    console.log(`⚖️ Rebalance opportunity for ${symbol}: $${shortfall.toFixed(2)} shortfall`);
                    
                    // Calculate buy quantity
                    const buyQuantity = shortfall / row.Price;
                    const roundedQty = this._roundQuantity(symbol, buyQuantity);
                    
                    if (roundedQty > 0) {
                        console.log(`📉 Buying ${roundedQty} ${symbol} (~$${shortfall.toFixed(2)})`);
                        
                        // Execute buy on first available exchange
                        let success = false;
                        if (this.api && this.api.isWsConnected()) {
                            try {
                                const result = await this.api.placeBuy(symbol, roundedQty);
                                console.log(`✅ Kraken buy executed: ${result.id}`);
                                success = true;
                                this.balance -= shortfall;
                            } catch (error) {
                                console.error("❌ Kraken buy failed:", error.message);
                            }
                        }
                        
                        if (!success && this.coinbaseApi) {
                            try {
                                const result = await this.coinbaseApi.placeBuy(symbol, roundedQty);
                                console.log(`✅ Coinbase buy executed: ${result.id}`);
                                success = true;
                                this.balance -= shortfall;
                            } catch (error) {
                                console.error("❌ Coinbase buy failed:", error.message);
                            }
                        }
                        
                        if (success) {
                            // Update baseline
                            this.state.baselines[symbol] = baseline * (1 - config.TARGET_ADJUST_PERCENT);
                            this._saveState();
                        }
                    }
                }
            }
        }
    }
    
    _roundQuantity(symbol, quantity) {
        // Try Kraken first, then Coinbase, then default
        if (this.api && this.api.roundQty) {
            const result = this.api.roundQty(symbol, quantity);
            return parseFloat(result);
        }
        
        if (this.coinbaseApi && this.coinbaseApi.roundQty) {
            const result = this.coinbaseApi.roundQty(symbol, quantity);
            return parseFloat(result);
        }
        
        // Default rounding
        return Math.floor(quantity * 100000000) / 100000000;
    }
    
    _displayStatus() {
        console.log(`\n📈 STATUS:`);
        console.log(`  Exchange: ${this.exchange}`);
        console.log(`  Mode: ${this.mode}`);
        console.log(`  Balance: $${this.balance.toFixed(2)}`);
        console.log(`  Holdings: ${Array.from(this.holdings.keys()).join(', ') || 'None'}`);
        
        const activeBaselines = Object.keys(this.state.baselines).length;
        console.log(`  Active Baselines: ${activeBaselines}`);
    }
    
    _loadState() {
        try {
            const stateFile = path.join(process.cwd(), 'unifiedTradingState.json');
            if (fs.existsSync(stateFile)) {
                const data = fs.readFileSync(stateFile, 'utf-8');
                const state = JSON.parse(data);
                this.state = state;
                console.log(`✅ Loaded state from ${stateFile}`);
            }
        } catch (error) {
            console.warn("⚠️  Could not load state:", error.message);
        }
    }
    
    _saveState() {
        try {
            const stateFile = path.join(process.cwd(), 'unifiedTradingState.json');
            fs.writeFileSync(stateFile, JSON.stringify(this.state, null, 2));
        } catch (error) {
            console.error("❌ Could not save state:", error.message);
        }
    }
    
    _wait(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
    
    stop() {
        this.running = false;
    }
}

// ============== Main Application ==============
async function main() {
    console.log("🎯 Unified Trading System - Kraken + Coinbase + Backtesting");
    console.log("=".repeat(60));
    
    // Parse command line arguments
    const args = process.argv.slice(2);
    let mode = 'simulation';
    let exchange = 'kraken';
    
    if (args.includes('--backtest')) {
        mode = 'backtest';
        console.log("🔧 Running in BACKTEST mode");
    } else if (args.includes('--live')) {
        mode = 'live';
        console.log("⚠️  Running in LIVE mode (not yet fully implemented)");
    } else {
        console.log("🔧 Running in SIMULATION mode");
    }
    
    if (args.includes('--coinbase')) {
        exchange = 'coinbase';
        console.log("💵 Using Coinbase exchange");
    } else if (args.includes('--both')) {
        exchange = 'both';
        console.log("💵 Using both Kraken and Coinbase");
    } else {
        console.log("💵 Using Kraken exchange");
    }
    
    // Create and initialize system
    const system = new UnifiedTradingSystem(mode, exchange);
    
    try {
        await system.initialize();
        await system.run();
    } catch (error) {
        console.error("💥 FATAL ERROR:", error.message);
        if (error.stack) {
            console.error(error.stack);
        }
    }
    
    console.log("✅ Unified Trading System completed");
}

// Run the application
if (import.meta.url === `file://${process.argv[1]}`) {
    main().catch(console.error);
}

// Export for testing
export {
    UnifiedTradingSystem,
    KrakenAPI,
    CoinbaseAPI,
    BacktestingEngine,
    UNIFIED_CONFIG
};