// cryptobot-kraken-baseline.js (ESM style) - v2.4.1-API-Merged-Fix2
// *** MODIFIED: Replaced kraken-api client with enhanced_bot.js KrakenAPI class ***
// *** MODIFIED: Quote fetching now uses WebSockets via api.getLatestPrice() ***
// *** MODIFIED: Quantity rounding uses api.roundQty() ***
// *** MODIFIED: Min order quantity fetched via new api.getPairData() method ***
// *** FIX V2.4.1-API-Merged-Fix1: Prevent initial WebSocket price check from resetting valid loaded baselines during startup verification. Loaded baselines are now trusted if found. ***
// *** FIX V2.4.1-API-Merged-Fix2: Fixed ReferenceError for totalReinvestedThisCycle by declaring it in the correct scope within harvest allocation logic. ***
// - All original strategy logic (v2.4.1 - Adaptive DZ, Revised CP, Harvest, Rebalance, Persistence) remains unchanged.
// - All original console logging within mainLoop remains unchanged (except for baseline init log).
// - ADZ activation logic confirmed correct via isolated testing; previous discrepancies attributed to price source differences (WebSocket vs REST) affecting boundary checks.
// -------------------------------------------------------------------

// ---- CORE IMPORTS ----
import dotenv from 'dotenv';
import { Buffer } from 'buffer';
import crypto from 'crypto';
import readline from 'readline';
import fs from 'fs';
import path from 'path';

// ---- IMPORTS FOR NEW API ----
import WebSocket from 'ws';
const fetch = global.fetch || (await import('node-fetch')).default;
import chalk from 'chalk';
import sparkly from 'sparkly'; // Note: sparkly/table not directly used by user logic, but are deps of the API class
import { table } from 'console';

// ---- IMPORT THE LOGGER ----
import { appendTradeHistory } from './tradeHistory.js'; // Assuming this exists and works

dotenv.config();

// ============== Config/Maps and Constants (User's Original Section) ==============
// --- Kraken Specific ---
const INITIAL_KRAKEN_ASSET_MAP = {
    'XXBT': 'BTC', 'ETH': 'ETH', 'USDC': 'USDC', 'SOL': 'SOL', 'ADA': 'ADA', 'XLM': 'XLM', 'AVAX': 'AVAX', 'XRP': 'XRP', 'LINK': 'LINK', 'UNI': 'UNI', 'DOGE': 'DOGE', 'SHIB': 'SHIB', 'PEPE': 'PEPE', 'BONK': 'BONK', 'WIF': 'WIF', 'AAVE': 'AAVE', 'COMP': 'COMP', 'POPCAT': 'POPCAT', 'ALGO': 'ALGO', 'FET': 'FET', 'ICP': 'ICP', 'NEAR': 'NEAR', 'RENDER': 'RENDER', 'TAO': 'TAO', 'XLTC': 'LTC', 'TRX': 'TRX', 'SUI': 'SUI', 'XXDG': 'XDG', 'ZUSD': 'USD', 'INJ': 'INJ', 'OCEAN': 'OCEAN', 'TRUMP': 'TRUMP'
};
const QUOTE_CURRENCY = 'USD';

// --- Exclusions (Use Common Symbols) ---
const HARVEST_EXCLUDE = ["BTC", "USDC", QUOTE_CURRENCY];
const REBALANCE_EXCLUDE = ["BTC", "USDC", QUOTE_CURRENCY];

// --- Core Strategy ---
const TARGET_ADJUST_PERCENT = 0.000;

// --- Individual Asset Harvest ---
const FLAT_HARVEST_TRIGGER_PERCENT = 0.03;
const HARVEST_CYCLE_THRESHOLD = 3;
const MIN_SURPLUS_FOR_HARVEST = 1.00;
const MIN_SURPLUS_FOR_FORCED_HARVEST = 1.00;
const FORCED_HARVEST_TIMEOUT = 20 * 60 * 1000;

// --- Portfolio Override Harvest (Baseline Reset) ---
const ENABLE_PORTFOLIO_HARVEST = true;
const PORTFOLIO_HARVEST_TRIGGER_DEVIATION_PERCENT = 0.05;
const PORTFOLIO_HARVEST_CONFIRMATION_CYCLES = 3;
const MIN_ASSET_SURPLUS_FOR_PORTFOLIO_HARVEST = 0.10;

// --- Harvest Proceeds Allocation ---
const HARVEST_ALLOC_BTC_PERCENT = 0.10;
const HARVEST_ALLOC_REINVEST_PERCENT = 0.50;
const HARVEST_ALLOC_CASH_PERCENT = 0.40; // Note: 0.10 + 0.50 + 0.40 = 1.00
const MIN_HARVEST_TO_ALLOCATE = 1.00;
const MIN_NEGATIVE_DEVIATION_FOR_REINVEST = -0.01;
const MIN_REINVEST_BUY_USD = 0.50;
const MIN_BTC_BUY_USD = 9999.10; // Keep BTC effectively disabled

// --- Rebalance ---
const FLAT_REBALANCE_TRIGGER_PERCENT = 0.04;
const PARTIAL_RECOVERY_PERCENT = 0.875;
const REBALANCE_POSITIVE_THRESHOLD = 3;
const MAX_REBALANCE_ATTEMPTS = 3;
const REBALANCE_COOLDOWN = 30 * 60 * 1000;
const FORCE_REBALANCE_TIMEOUT = 25 * 60 * 1000;
const FORCE_REBALANCE_SHORTFALL_PERCENT = 0.25;
const MIN_PARTIAL_REBALANCE_USD = 1.00;
const MIN_FORCED_REBALANCE_USD = 1.00;

// --- Adaptive Dead Zone Mode ---
const ENABLE_ADAPTIVE_DEAD_ZONE = true;
const ADAPTIVE_DZ_INACTIVITY_TIMEOUT = 3 * 60 * 60 * 1000; // 3 hours
const ADAPTIVE_DZ_HARVEST_TRIGGER_PERCENT = 0.020; // +2.0%
const ADAPTIVE_DZ_REBALANCE_TRIGGER_PERCENT = 0.020; // -2.0%
// Note: Requires +1 confirmation cycle & skips baseline adjustment (handled in logic)

// --- Portfolio-Level Crash Protection ---
const ENABLE_CRASH_PROTECTION = true;
const CP_TRIGGER_ASSET_PERCENT = 0.70;
const CP_TRIGGER_MIN_NEGATIVE_DEV_PERCENT = -0.01; // -1%
// --- CP Effects ---
const CRASH_PROTECTION_THRESHOLD_INCREASE = 2;
const CRASH_PROTECTION_PARTIAL_RECOVERY_PERCENT = 0.55;

// --- Timing & Persistence ---
const REFRESH_INTERVAL = 8000;
const STATE_FILE_PATH = path.join(process.cwd(), 'krakenBotState.json');
const BASELINE_LOAD_TOLERANCE_PERCENT = 0.15; // Kept for potential future use or logging, but not used for resetting loaded baselines anymore.

// ==================================================
// Sanity check allocation percentages
if (Math.abs(HARVEST_ALLOC_BTC_PERCENT + HARVEST_ALLOC_REINVEST_PERCENT + HARVEST_ALLOC_CASH_PERCENT - 1.0) > 0.001) {
    console.warn("Configuration Warning: Harvest allocation percentages do not sum precisely to 1.0 (100%).");
}
// ==================================================

// ============== Global State ==============
let tokenBaselines = {};
let trailingState = {};
let lastActionTimestamps = {};
let rebalanceState = {};
let portfolioHarvestState = {
    flagged: false, cycleCount: 0, previousDeviationPercent: null, flaggedAt: null
};
let adaptiveDeadZoneState = {};
let harvestedAmount = 0;
let initialized = false;
let assetMinOrderQuantities = {};
let api; // Declare api globally within the module scope for graceful shutdown

// ============== Kraken API Wrapper (NEW - Copied from enhanced_bot.js) ==============

// --- Constants FOR NEW API ---
const KRAKEN_ASSET_MAP = { ...INITIAL_KRAKEN_ASSET_MAP };
KRAKEN_ASSET_MAP['XBT'] = KRAKEN_ASSET_MAP['XBT'] ?? KRAKEN_ASSET_MAP['XXBT'] ?? 'BTC';
KRAKEN_ASSET_MAP['XXBT'] = KRAKEN_ASSET_MAP['XXBT'] ?? KRAKEN_ASSET_MAP['XBT'] ?? 'BTC';
KRAKEN_ASSET_MAP['XETH'] = KRAKEN_ASSET_MAP['XETH'] ?? KRAKEN_ASSET_MAP['ETH'] ?? 'ETH';
KRAKEN_ASSET_MAP['XXRP'] = KRAKEN_ASSET_MAP['XXRP'] ?? KRAKEN_ASSET_MAP['XRP'] ?? 'XRP';
KRAKEN_ASSET_MAP['ZUSD'] = KRAKEN_ASSET_MAP['ZUSD'] ?? QUOTE_CURRENCY;
KRAKEN_ASSET_MAP['USD'] = KRAKEN_ASSET_MAP['USD'] ?? QUOTE_CURRENCY;

const SYMBOL_TO_KRAKEN_ASSET = Object.fromEntries(
    Object.entries(KRAKEN_ASSET_MAP).map(([k, v]) => [v, k === 'XBT' ? 'XBT' : k])
);
const KRAKEN_QUOTE_ASSET = Object.entries(INITIAL_KRAKEN_ASSET_MAP).find(([_, v]) => v === QUOTE_CURRENCY)?.[0] || 'ZUSD';
SYMBOL_TO_KRAKEN_ASSET[QUOTE_CURRENCY] = KRAKEN_QUOTE_ASSET;

// --- API Internal Timings/Config ---
const API_INTERNAL_RETRY_DELAY_MS = 45000;
const API_INTERNAL_REQUEST_TIMEOUT_MS = 20000;
const WS_RECONNECT_DELAY_MS = 5000;
const MIN_ORDER_DUST_THRESHOLD = 1e-12;

// --- Logging Helpers FOR NEW API ---
const logPrefix = chalk.gray('[API]');
const logApiInfo = (...args) => console.log(logPrefix, ...args);
const logApiWarn = (...args) => console.warn(logPrefix, chalk.yellow('⚠️', ...args));
const logApiError = (...args) => console.error(logPrefix, chalk.redBright('❌', ...args));

// --- NEW KrakenAPI Class Definition ---
class KrakenAPI {
    #apiKey; #apiSecret; #apiUrl = 'https://api.kraken.com'; #wsPublicUrl = 'wss://ws.kraken.com';
    #wsAuthUrl = 'wss://ws-auth.kraken.com'; #retryDelay = API_INTERNAL_RETRY_DELAY_MS; #requestTimeout = API_INTERNAL_REQUEST_TIMEOUT_MS;
    #lastNonce = 0; secretBuffer; #pairInfo = new Map(); #assetInfo = new Map(); #pairNameToWsName = new Map();
    #wsNameToPairName = new Map(); #krakenAssetToCommonSymbol = new Map(); #wsPublic = null; #wsAuth = null; #wsToken = null;
    #wsTokenExpires = 0; #wsPublicConnected = false; #wsAuthConnected = false; #wsAuthenticated = false; #publicWsReconnectTimer = null;
    #authWsReconnectTimer = null; #latestPrices = new Map(); #activeSubscriptions = new Set();
    log; warn; error;

    constructor(apiKey, apiSecret) {
        if (!apiKey || !apiSecret) { throw new Error("Kraken API Key and Secret are required."); }
        this.#apiKey = apiKey; this.#apiSecret = apiSecret; this.secretBuffer = Buffer.from(this.#apiSecret, 'base64');
        this.log = logApiInfo; this.warn = logApiWarn; this.error = logApiError;
        this.#lastNonce = Date.now() * 1000;
    }
    _getNonce() { this.#lastNonce = Math.max(this.#lastNonce + 1, Date.now() * 1000); return this.#lastNonce; }
    _signRequest(path, requestBodyString, nonce) { try { const hash = crypto.createHash('sha256'); const hmac = crypto.createHmac('sha512', this.secretBuffer); const hashDigest = hash.update(nonce.toString() + requestBodyString).digest('binary'); const pathBuffer = Buffer.from(path); const hmacInput = Buffer.concat([pathBuffer, Buffer.from(hashDigest, 'binary')]); return hmac.update(hmacInput).digest('base64'); } catch (e) { this.error("Error during request signing:", e); throw new Error("Failed to sign request."); } }
    async _request(endpoint, params = {}, isPrivate = false, method = 'POST') {
        const path = `/0/${isPrivate ? 'private' : 'public'}/${endpoint}`; const url = this.#apiUrl + path; const headers = { 'User-Agent': 'KrakenBaselineBot-Node/2.4.1-API-Merged-Fix2' }; let bodyParams = { ...params }; let bodyString = ''; let bodyParamsJSON = ''; let nonce;
        if (isPrivate) { nonce = this._getNonce(); bodyParams.nonce = nonce; bodyString = new URLSearchParams(bodyParams).toString(); headers['API-Key'] = this.#apiKey; headers['API-Sign'] = this._signRequest(path, bodyString, nonce); headers['Content-Type'] = 'application/x-www-form-urlencoded; charset=utf-8'; } else if (method !== 'GET') { bodyString = new URLSearchParams(bodyParams).toString(); headers['Content-Type'] = 'application/x-www-form-urlencoded; charset=utf-8'; }
        const controller = new AbortController(); const timeoutId = setTimeout(() => controller.abort(), this.#requestTimeout);
        try {
            const response = await fetch(url, { method: method, headers: headers, body: (method !== 'GET' && bodyString) ? bodyString : undefined, signal: controller.signal }); clearTimeout(timeoutId);
            const responseBody = await response.text();
            if (!response.ok) { this.error(`HTTP Error ${response.status} on ${method} ${url}. Body: ${responseBody}`); throw new Error(`HTTP Error ${response.status}: ${response.statusText}`); }
            const data = JSON.parse(responseBody);
            if (data.error && data.error.length > 0) { const errorMsg = data.error.join(', '); const krakenError = new Error(`Kraken API Error (${endpoint}): ${errorMsg}`); krakenError.krakenErrors = data.error; if (errorMsg.includes("EAPI:Invalid nonce")) { this.error(`>>> Nonce error detected! Nonce sent: ${nonce}. Last recorded nonce: ${this.#lastNonce}.`); } throw krakenError; }
            return data.result;
        } catch (error) { clearTimeout(timeoutId); if (error.name === 'AbortError') { throw new Error(`Request (${endpoint}) timed out after ${this.#requestTimeout / 1000}s`); } this.error(`Error during _request for ${endpoint}:`, error); throw error; }
    }
    async _requestWithRetry(endpoint, params = {}, isPrivate = false, method = 'POST') {
        let attempts = 0; const maxAttempts = 5;
        while (attempts < maxAttempts) {
            attempts++;
            try { return await this._request(endpoint, params, isPrivate, method); } catch (error) {
                this.warn(`Request Warn (${endpoint}): Attempt ${attempts}/${maxAttempts}: ${error.message}`); const krakenErrors = error.krakenErrors || []; const isNetworkError = !error.krakenErrors && !error.message?.includes('timed out') && !error.message?.startsWith('HTTP Error'); const isRetryableHttp = error.message?.startsWith('HTTP Error 5'); const isRateLimit = krakenErrors.some(e => /Rate limit exceeded/i.test(e)); const isTempKrakenError = krakenErrors.some(e => /^(EService:|EGeneral:Temporary|EQuery:Unknown|EAPI:Invalid nonce)/i.test(e));
                if ((isNetworkError || isRetryableHttp || isRateLimit || isTempKrakenError) && attempts < maxAttempts) { this.log(`Retryable error detected. Retrying in ${this.#retryDelay / 1000}s...`); await new Promise((res) => setTimeout(res, this.#retryDelay)); continue; } else { this.error(`API Error (${endpoint}): Non-retryable or max attempts reached.`); throw error; }
            }
        } throw new Error(`API request failed after ${maxAttempts} attempts for ${endpoint}.`);
    }
    async initialize() {
        this.log("Initializing Kraken API client (Enhanced Version)..."); try { const [assetResult, pairResult] = await Promise.all([this._requestWithRetry('Assets', {}, false, 'GET'), this._requestWithRetry('AssetPairs', {}, false, 'GET')]); this.#assetInfo = new Map(Object.entries(assetResult)); this.#pairInfo = new Map(Object.entries(pairResult)); this.#pairNameToWsName.clear(); this.#wsNameToPairName.clear(); this.#krakenAssetToCommonSymbol.clear();
        for (const [krakenAsset, commonSymbol] of Object.entries(KRAKEN_ASSET_MAP)) { this.#krakenAssetToCommonSymbol.set(krakenAsset, commonSymbol); }
        for (const [krakenAsset, data] of this.#assetInfo.entries()) { if (!this.#krakenAssetToCommonSymbol.has(krakenAsset) && data.altname) { const common = KRAKEN_ASSET_MAP[data.altname]; if(common) { this.#krakenAssetToCommonSymbol.set(krakenAsset, common); } } }
        if (!this.#krakenAssetToCommonSymbol.has(KRAKEN_QUOTE_ASSET)) { this.#krakenAssetToCommonSymbol.set(KRAKEN_QUOTE_ASSET, QUOTE_CURRENCY); this.warn(`Manually mapped ${KRAKEN_QUOTE_ASSET} to ${QUOTE_CURRENCY}`); }
        for (const [pairName, data] of this.#pairInfo.entries()) { if (data.wsname) { this.#pairNameToWsName.set(pairName, data.wsname); this.#wsNameToPairName.set(data.wsname, pairName); } }
        this.log(chalk.green(`Fetched info for ${this.#assetInfo.size} assets and ${this.#pairInfo.size} pairs.`)); this._connectWebsockets(); this.log("WebSocket connections initiated..."); await new Promise(resolve => setTimeout(resolve, 5000));
        } catch (error) { this.error("FATAL: Failed to initialize API.", error); throw error; }
    }
    async #getWsToken() { if (this.#wsToken && Date.now() < this.#wsTokenExpires * 0.95) { return this.#wsToken; } this.log("Fetching new WebSocket authentication token..."); try { const result = await this._requestWithRetry('GetWebSocketsToken', {}, true, 'POST'); this.#wsToken = result.token; this.#wsTokenExpires = Date.now() + (parseInt(result.expires) * 1000); this.log(chalk.green("WebSocket token obtained.")); return this.#wsToken; } catch (error) { this.error("Failed to get WebSocket token:", error); this.#wsToken = null; this.#wsTokenExpires = 0; throw error; } }
    _connectWebsockets() { this._connectPublicWs(); /* this._connectAuthWs(); */ }
    _connectPublicWs() {
        if (this.#publicWsReconnectTimer) clearTimeout(this.#publicWsReconnectTimer); if (this.#wsPublic && [WebSocket.CONNECTING, WebSocket.OPEN].includes(this.#wsPublic.readyState)) { return; }
        this.log(`🔌 Connecting to Public WebSocket: ${this.#wsPublicUrl}`); this.#wsPublic = new WebSocket(this.#wsPublicUrl); this.#wsPublicConnected = false;
        this.#wsPublic.on('open', () => { this.log(chalk.green('✅ Public WebSocket Connected.')); this.#wsPublicConnected = true; if (this.#activeSubscriptions.size > 0) { this.log(`Resubscribing to ${this.#activeSubscriptions.size} pairs on public WS reconnect...`); this._sendWsSubscription([...this.#activeSubscriptions]); } });
        this.#wsPublic.on('message', (data) => this._handlePublicMessage(data)); this.#wsPublic.on('error', (err) => this.error('Public WebSocket Error:', err.message));
        this.#wsPublic.on('close', (code, reason) => { this.#wsPublicConnected = false; this.warn(`Public WebSocket Closed. Code: ${code}, Reason: ${reason?.toString()}. Reconnecting in ${WS_RECONNECT_DELAY_MS / 1000}s...`); this.#activeSubscriptions.clear(); this.#latestPrices.clear(); /* Clear prices on disconnect */ this.#publicWsReconnectTimer = setTimeout(() => this._connectPublicWs(), WS_RECONNECT_DELAY_MS); });
    }
    _connectAuthWs() { if (this.#authWsReconnectTimer) clearTimeout(this.#authWsReconnectTimer); if (this.#wsAuth && [WebSocket.CONNECTING, WebSocket.OPEN].includes(this.#wsAuth.readyState)) { return; } this.log(`🔌 Connecting to Authenticated WebSocket: ${this.#wsAuthUrl}`); this.#wsAuth = new WebSocket(this.#wsAuthUrl); this.#wsAuthConnected = false; this.#wsAuthenticated = false; this.#wsAuth.on('open', async () => { /* ... Auth logic ... */ }); this.#wsAuth.on('message', (data) => this._handleAuthMessage(data)); this.#wsAuth.on('error', (err) => this.error('Authenticated WebSocket Error:', err.message)); this.#wsAuth.on('close', (code, reason) => { /* ... Reconnect logic ... */ }); }
    _handlePublicMessage(data) {
        try {
            const message = JSON.parse(data.toString()); if (message.event === 'heartbeat') return; if (message.event === 'systemStatus') { this.log(`Public WS System Status: ${chalk.cyan(message.status?.toUpperCase() ?? 'UNKNOWN')}`); return; }
            if (message.event === 'subscriptionStatus') { const { pair, status, subscription, errorMessage, channelName, reqid } = message; const pairWsName = pair ?? channelName?.split('-')[1] ?? 'N/A'; const statusStr = status?.toUpperCase() ?? 'UNKNOWN'; const subName = subscription?.name ?? channelName?.split('-')[0] ?? 'N/A'; const reqidStr = reqid ? ` (reqid: ${reqid})` : ''; if (status === 'error') { this.error(`Subscription Error for ${pairWsName}: ${errorMessage ?? 'No error message'}${reqidStr}`); if (pair) this.#activeSubscriptions.delete(pair); } else if (status === 'unsubscribed') { this.log(`Public WS Unsubscribed: Pair ${pairWsName} - ${subName}${reqidStr}`); if (pair) this.#activeSubscriptions.delete(pair); } else if (status === 'subscribed') { this.log(chalk.green(`Public WS Subscribed: Pair ${pairWsName} - ${subName}${reqidStr}`)); if (pair) this.#activeSubscriptions.add(pair); } else { this.log(`Public WS Subscription Update: Pair ${pairWsName} Status ${statusStr} - ${subName}${reqidStr}`); } return; }
            if (Array.isArray(message) && message.length >= 4 && message[2] === 'ticker' && typeof message[3] === 'string') { const pairWsName = message[3]; const priceStr = message[1]?.c?.[0]; const price = parseFloat(priceStr); if (!isNaN(price) && price > 0) { this.#latestPrices.set(pairWsName, price); } else { /* Keep existing price if new one is invalid? Or delete? Deleting for now. */ this.#latestPrices.delete(pairWsName); this.warn(`Received invalid ticker price for ${pairWsName}: ${priceStr}`); } }
        } catch (error) { this.error('Error processing Public WS message:', error, data.toString()); }
    }
    _handleAuthMessage(data) { /* Defined for potential future use */ }
    _sendWsSubscription(krakenPairWsNames, unsubscribe = false) { if (!this.#wsPublic || this.#wsPublic.readyState !== WebSocket.OPEN) { this.warn(`Cannot ${unsubscribe ? 'unsubscribe' : 'subscribe'}: Public WebSocket not connected/ready.`); return; } if (!Array.isArray(krakenPairWsNames) || krakenPairWsNames.length === 0) return; const event = unsubscribe ? 'unsubscribe' : 'subscribe'; const message = { event, pair: krakenPairWsNames, subscription: { name: 'ticker' } }; try { this.#wsPublic.send(JSON.stringify(message)); } catch (error) { this.error(`Failed to send WS message: ${error.message}`); } }
    isWsConnected() { return this.#wsPublicConnected; }
    async getHoldings() {
        try { const balances = await this._requestWithRetry('Balance', {}, true); const holdingsMap = new Map(); let quoteBalance = 0;
        for (const [krakenAsset, qtyStr] of Object.entries(balances)) { const quantity = parseFloat(qtyStr); const commonSymbol = this.#krakenAssetToCommonSymbol.get(krakenAsset); if (krakenAsset === KRAKEN_QUOTE_ASSET) { if (!isNaN(quantity)) { quoteBalance += quantity; } } else if (commonSymbol === QUOTE_CURRENCY) { if (!isNaN(quantity)) { quoteBalance += quantity; } } else if (commonSymbol && quantity > MIN_ORDER_DUST_THRESHOLD) { if (holdingsMap.has(commonSymbol)) { const existing = holdingsMap.get(commonSymbol); existing.total_quantity += quantity; existing.kraken_assets.push(krakenAsset); } else { holdingsMap.set(commonSymbol, { asset_code: commonSymbol, total_quantity: quantity, kraken_assets: [krakenAsset] }); } } else if (quantity > MIN_ORDER_DUST_THRESHOLD) { /* Warn unmapped */ this.warn(`Unmapped asset ${krakenAsset} with quantity ${quantity} found.`); } }
        quoteBalance = isNaN(quoteBalance) ? 0 : quoteBalance; const holdingsArray = Array.from(holdingsMap.values()); return [quoteBalance, holdingsArray];
        } catch (error) { this.error("Error fetching holdings:", error.message); throw error; }
    }
    getKrakenPairName(commonSymbol) {
        const krakenBaseAsset = SYMBOL_TO_KRAKEN_ASSET[commonSymbol]; const btcVariants = ['XBT', 'XXBT']; const baseAssetsToTry = commonSymbol === 'BTC' ? [...new Set([krakenBaseAsset, ...btcVariants].filter(Boolean))] : [krakenBaseAsset || commonSymbol]; const targetKrakenQuoteAsset = KRAKEN_QUOTE_ASSET;
        for (const base of baseAssetsToTry) { for (const [pairKey, pairData] of this.#pairInfo.entries()) { if (pairData.base === base && pairData.quote === targetKrakenQuoteAsset) { return pairKey; } if (pairKey === `${base}${targetKrakenQuoteAsset}`) { /* Less reliable check */ if(this.#pairInfo.has(pairKey)) return pairKey; } } }
        for (const base of baseAssetsToTry) { const pairGuess = `${base}${QUOTE_CURRENCY}`; if (this.#pairInfo.has(pairGuess)) { return pairGuess; } } // Final guess
        this.warn(`Could not find valid Kraken pair name for symbol ${commonSymbol} (Tried bases: ${baseAssetsToTry.join('/')}, Quote: ${targetKrakenQuoteAsset}/${QUOTE_CURRENCY})`); return null;
    }
    getPairData(commonSymbol) { const pairName = this.getKrakenPairName(commonSymbol); if (!pairName) { return null; } const pairData = this.#pairInfo.get(pairName); if (!pairData) { return null; } return pairData; }
    getKrakenPairWsName(commonSymbol) { const pairName = this.getKrakenPairName(commonSymbol); if (!pairName) return null; return this.#pairNameToWsName.get(pairName) ?? null; }
    getLatestPrice(commonSymbol) { const pairWsName = this.getKrakenPairWsName(commonSymbol); if (!pairWsName) { return undefined; } return this.#wsPublicConnected ? this.#latestPrices.get(pairWsName) : undefined; }
    subscribeToTickers(commonSymbols) {
        if (!this.isWsConnected()) { this.warn("WebSocket not connected, subscription deferred."); return; } const targetWsPairs = new Set();
        for (const sym of commonSymbols) { const wsName = this.getKrakenPairWsName(sym); if (wsName) { targetWsPairs.add(wsName); } else { this.warn(`Cannot subscribe ticker: No WebSocket pair name found for ${sym}`); } }
        const currentSubs = new Set(this.#activeSubscriptions); const pairsToSubscribe = [...targetWsPairs].filter(p => !currentSubs.has(p)); const pairsToUnsubscribe = [...currentSubs].filter(p => !targetWsPairs.has(p));
        if (pairsToUnsubscribe.length > 0) this._sendWsSubscription(pairsToUnsubscribe, true); if (pairsToSubscribe.length > 0) this._sendWsSubscription(pairsToSubscribe, false);
    }
    roundQty(commonSymbol, quantity) {
        if (isNaN(quantity) || quantity <= MIN_ORDER_DUST_THRESHOLD) return "0.0"; const pairName = this.getKrakenPairName(commonSymbol); if (!pairName) { this.error(`Cannot round qty for ${commonSymbol}: No Kraken pair name found.`); return "0.0"; } const pairData = this.#pairInfo.get(pairName); if (!pairData) { this.error(`Cannot round qty for ${commonSymbol}: Missing pair info for ${pairName}.`); return "0.0"; }
        const lotDecimals = pairData.lot_decimals; const minOrderVolumeStr = pairData.ordermin; if (typeof lotDecimals !== 'number' || lotDecimals < 0 || minOrderVolumeStr === undefined) { this.error(`Cannot round qty for ${pairName}: Invalid pair data (decimals: ${lotDecimals}, ordermin: ${minOrderVolumeStr})`); return "0.0"; } const minOrderVolume = parseFloat(minOrderVolumeStr); if (isNaN(minOrderVolume) || minOrderVolume < 0) { this.error(`Cannot round qty for ${pairName}: Invalid minOrderVolume '${minOrderVolumeStr}'`); return "0.0"; }
        const epsilon = 1e-12; if (quantity < (minOrderVolume - epsilon)) { return "0.0"; } const factor = Math.pow(10, lotDecimals); const roundedQty = Math.floor(quantity * factor) / factor; if (roundedQty < (minOrderVolume - epsilon)) { return "0.0"; } if (roundedQty <= MIN_ORDER_DUST_THRESHOLD) { return "0.0"; } return roundedQty.toFixed(lotDecimals);
    }
    async placeSell(commonSymbol, quantityStr) { const pairName = this.getKrakenPairName(commonSymbol); if (!pairName) throw new Error(`Cannot place sell: Invalid Kraken pair for ${commonSymbol}`); return this._placeOrderInternal(pairName, 'sell', quantityStr); }
    async placeBuy(commonSymbol, quantityStr) { const pairName = this.getKrakenPairName(commonSymbol); if (!pairName) throw new Error(`Cannot place buy: Invalid Kraken pair for ${commonSymbol}`); return this._placeOrderInternal(pairName, 'buy', quantityStr); }
    async _placeOrderInternal(krakenPairName, side, quantityStr) {
        const quantity = parseFloat(quantityStr); if (isNaN(quantity) || quantity <= 0) { throw new Error(`Invalid quantity for ${side} order: '${quantityStr}'`); }
        try { const orderDetails = { pair: krakenPairName, type: side, ordertype: 'market', volume: quantityStr }; this.log(chalk.cyan(`Attempting ${side.toUpperCase()} order: ${quantityStr} ${krakenPairName}`)); const responseResult = await this._requestWithRetry('AddOrder', orderDetails, true, 'POST'); const orderId = responseResult?.txid?.[0] ?? `local_order_${crypto.randomUUID()}`; const description = responseResult?.descr?.order ?? 'No description'; this.log(chalk.green(`✅ ${side.toUpperCase()} Order Submitted (${krakenPairName}): ${description}. ID: ${orderId}`)); return { id: orderId, client_order_id: orderId, description: description };
        } catch (error) { this.error(`Failed to place ${side.toUpperCase()} order for ${quantityStr} ${krakenPairName}:`, error.message); if (error.krakenErrors) { this.error("Kraken specific errors:", error.krakenErrors.join(', ')); } throw error; }
    }
    close() { this.log("Closing API connections..."); clearTimeout(this.#publicWsReconnectTimer); clearTimeout(this.#authWsReconnectTimer); const closeWs = (ws, name) => { if (ws) { this.log(`Closing ${name} WS...`); try { ws.removeAllListeners(); ws.terminate(); } catch (e) { this.warn(`Error terminating ${name} WS: ${e.message}`);} } }; closeWs(this.#wsPublic, 'Public'); closeWs(this.#wsAuth, 'Auth'); this.#wsPublicConnected = false; this.#wsAuthConnected = false; this.#wsAuthenticated = false; this.#activeSubscriptions.clear(); this.#latestPrices.clear(); this.log("API connections closed."); }
} // End NEW KrakenAPI Class

// ============== Persistence Functions ==============
function loadState() {
    let loadedBaselines = {}; let loadedTrailingState = {}; let loadedLastActionTimestamps = {};
    try {
        if (fs.existsSync(STATE_FILE_PATH)) { const data = fs.readFileSync(STATE_FILE_PATH, 'utf-8'); const loadedData = JSON.parse(data); loadedBaselines = loadedData.baselines || {}; loadedTrailingState = loadedData.trailingState || {}; loadedLastActionTimestamps = loadedData.lastActionTimestamps || {}; console.log(`✅ Loaded state (Baselines, TrailingState, LastActionTimestamps) from ${STATE_FILE_PATH}.`); }
        else { console.log(`ℹ️ State file ${STATE_FILE_PATH} not found, starting fresh.`); }
    } catch (err) { console.error(`❌ Error loading state from ${STATE_FILE_PATH}:`, err); }
    return { loadedBaselines, loadedTrailingState, loadedLastActionTimestamps };
}
function saveState() {
    try { const stateToSave = { baselines: tokenBaselines, trailingState: trailingState, lastActionTimestamps: lastActionTimestamps }; const tempFilePath = STATE_FILE_PATH + '.tmp'; fs.writeFileSync(tempFilePath, JSON.stringify(stateToSave, null, 2)); fs.renameSync(tempFilePath, STATE_FILE_PATH); }
    catch (err) { console.error("🚨 CRITICAL ERROR: Failed to save state:", err); }
}

// ============== Helper Functions ==============
function logTrade({ asset, side, quantity, price, clientOrderId, note = "" }) {
    try { const numericPrice = parseFloat(price); const numericQuantity = parseFloat(quantity); let totalValue = 'N/A'; if (!isNaN(numericPrice) && !isNaN(numericQuantity) && numericPrice > 0 && numericQuantity > 0) { totalValue = (numericPrice * numericQuantity).toFixed(2); } else { console.warn(`Error calc trade value ${asset}`); price = String(price); quantity = String(quantity); } appendTradeHistory({ asset, side: side.toUpperCase(), orderType: "market", quantity: quantity.toString(), effectivePrice: price.toString(), totalValue, clientOrderId: clientOrderId || 'N/A', extra: { note }, }); }
    catch (error) { console.error(`Error logging trade ${asset}:`, error); }
}
function getEffectivePriceFromResp(resp, fallbackPrice) { return fallbackPrice; } // Fallback remains necessary - API response doesn't give fill price
const wait = (ms) => new Promise(resolve => setTimeout(resolve, ms));

// ============== Main Application Logic ==============
const rl = readline.createInterface({ input: process.stdin, output: process.stdout });

async function mainLoop() {
    console.log("🚀 Initializing Kraken CryptoBot (v2.4.1-API-Merged-Fix2)...");

    const apiKey = process.env.KRAKEN_API_KEY;
    const apiSecret = process.env.KRAKEN_PRIVATE_KEY;
    if (!apiKey || !apiSecret) { console.error("❌ FATAL: Missing Kraken keys."); rl.close(); return; }

    // api declared globally now
    try { api = new KrakenAPI(apiKey, apiSecret); await api.initialize(); console.log("🔑 Kraken API Initialized (Enhanced Version)."); }
    catch (error) { console.error("❌ FATAL: Kraken init error:", error.message); rl.close(); return; }

    // --- Main Loop Start ---
    while (true) {
        let isGlobalRiskSignalActive = false; // Hoisted declaration
        const startTime = Date.now();
        console.log(`\n----- Cycle Start: ${new Date().toISOString()} -----`);
        harvestedAmount = 0; let anyTradesThisCycle = false; let stateChanged = false; assetMinOrderQuantities = {};

        // 1 & 2) Fetch Balance & Holdings
        let cashBalance = 0; let holdings = []; let holdingDetails = {}; let codes = [];
        try {
            [cashBalance, holdings] = await api.getHoldings();
            console.log(`💰 Available Balance: $${cashBalance.toFixed(2)} ${QUOTE_CURRENCY}`);
            if (holdings.length > 0) { holdings.forEach(h => { const code = h.asset_code; const qty = h.total_quantity; holdingDetails[code] = { rawQuantity: qty, krakenAssets: h.kraken_assets }; codes.push(code); }); codes.sort(); console.log(`📊 Holdings: ${codes.join(', ')}`); }
            else { console.log("ℹ️ No Holdings Found."); }
        } catch (err) { console.error("❌ ERROR: Balance/Holdings fetch failed:", err.message); await wait(REFRESH_INTERVAL); continue; }

        // 3) Subscribe to Tickers & Wait
        let requiredSymbols = [...codes];
        if (HARVEST_ALLOC_BTC_PERCENT > 0 && MIN_BTC_BUY_USD < 1000) { if (!requiredSymbols.includes('BTC')) { requiredSymbols.push('BTC'); } }
        requiredSymbols = [...new Set(requiredSymbols)];
        if (requiredSymbols.length > 0) { api.subscribeToTickers(requiredSymbols); await wait(500); } // Short wait for WS prices to potentially arrive

        // 4) Get Minimum Order Quantities
        if (codes.length > 0) {
            for (const sym of codes) {
                const pairData = api.getPairData(sym); if (!pairData) { continue; }
                const minOrderVolumeStr = pairData.ordermin;
                if (minOrderVolumeStr !== undefined) { const parsedMin = parseFloat(minOrderVolumeStr); if (!isNaN(parsedMin) && parsedMin > 0) { assetMinOrderQuantities[sym] = parsedMin; } else { console.warn(`[MinQty] Invalid ordermin value '${minOrderVolumeStr}' for ${sym}`); } }
                else { console.warn(`[MinQty] Missing ordermin field in pair data for ${sym}`); }
            }
        }

        // 5) Calculate Portfolio Summary, Initialize/Verify Baselines & State, Cleanup
        let totalHoldingsValue = 0; const portfolioSummary = []; const currentSymbols = new Set(); let baselinesVerifiedOrSetThisCycle = false; let priceFetchIssues = [];
        codes.forEach((sym) => {
            const details = holdingDetails[sym]; const price = api.getLatestPrice(sym);
            if (price === undefined || price === null || isNaN(price) || price <= 0) { priceFetchIssues.push(sym); if (trailingState[sym]) delete trailingState[sym].previousDeviation; if (rebalanceState[sym]) delete rebalanceState[sym].previousDeviation; if (adaptiveDeadZoneState[sym]) delete adaptiveDeadZoneState[sym]; return; }
            currentSymbols.add(sym); const totalQty = details.rawQuantity; const currentHoldingValue = price * totalQty; totalHoldingsValue += currentHoldingValue;

            // Baseline Init/Verify Logic (Fix1 applied)
            if (!initialized) {
                const loadedBaseline = tokenBaselines[sym];
                if (loadedBaseline !== undefined && typeof loadedBaseline === 'number' && loadedBaseline > 0.01) {
                    console.log(`✅ ${sym}: Using loaded baseline $${loadedBaseline.toFixed(2)}.`);
                    baselinesVerifiedOrSetThisCycle = true;
                } else if (tokenBaselines[sym] === undefined && currentHoldingValue > 0.01) {
                    tokenBaselines[sym] = currentHoldingValue;
                    console.log(`✨ Initialized baseline ${sym}: $${tokenBaselines[sym].toFixed(2)} (First cycle).`);
                    baselinesVerifiedOrSetThisCycle = true;
                    stateChanged = true;
                }
            }

            // Handle newly acquired assets AFTER initialization phase
            if (initialized && !tokenBaselines[sym] && currentHoldingValue > 0.01) {
                tokenBaselines[sym] = currentHoldingValue;
                console.log(`✨ Initialized baseline ${sym} (post-init): $${tokenBaselines[sym].toFixed(2)}.`);
                stateChanged = true;
            }

            // Initialize timestamp if missing for an asset with a baseline
            if (!lastActionTimestamps[sym] && tokenBaselines[sym] && tokenBaselines[sym] > 0.01) {
                console.log(`✨ Initialized last action timestamp for ${sym}.`);
                lastActionTimestamps[sym] = Date.now();
                stateChanged = true;
            }

            // Calculate deviation based on the (potentially just loaded or initialized) baseline
            const currentBaseline = tokenBaselines[sym];
            let deviation = NaN; let absoluteDifference = NaN;
            if (currentBaseline && typeof currentBaseline === 'number' && currentBaseline > 0) {
                deviation = (currentHoldingValue - currentBaseline) / currentBaseline;
                absoluteDifference = currentHoldingValue - currentBaseline;
            }
            portfolioSummary.push({ Symbol: sym, Quantity: totalQty, Price: price, Value: currentHoldingValue, Baseline: currentBaseline, Deviation: deviation, AbsoluteDifference: absoluteDifference, rawQuantity: totalQty, currentPrice: price, usdValueNum: currentHoldingValue, krakenAssets: details.krakenAssets.join(',') });
        }); // End forEach code

        if (priceFetchIssues.length > 0) { console.warn(`⚠️ Warn: Price unavailable/invalid via WebSocket for: [${priceFetchIssues.join(', ')}]. Calculations skipped for these assets.`); }

        // Mark initialization complete after the first pass if any baselines were handled
        if (!initialized && baselinesVerifiedOrSetThisCycle) {
            console.log("✅ Baselines & Timestamps init/verify complete.");
            initialized = true;
            if (stateChanged) { saveState(); stateChanged = false; } // Save state if baselines were initialized
        }
        else if (!initialized && holdings.length > 0 && codes.length === 0 && priceFetchIssues.length === holdings.length) { // Use holdings.length here
            console.log("⏳ Waiting for prices for baseline init...");
        }
        else if (!initialized && holdings.length === 0) {
            console.log("✅ No holdings, baseline init complete.");
            initialized = true; // Mark initialized even with no holdings
        }

        // State Cleanup (Original logic unchanged)
        let deletedKeys = false; Object.keys(tokenBaselines).forEach(sym => { if (!currentSymbols.has(sym)) { console.log(`🗑️ Clearing state for sold/removed asset: ${sym}.`); delete tokenBaselines[sym]; delete trailingState[sym]; delete rebalanceState[sym]; delete lastActionTimestamps[sym]; delete adaptiveDeadZoneState[sym]; deletedKeys = true; } }); Object.keys(trailingState).forEach(sym => { if (!tokenBaselines[sym] && currentSymbols.has(sym)) { console.log(`🗑️ Clearing trailing state ${sym} (no baseline).`); delete trailingState[sym]; deletedKeys = true; } }); Object.keys(lastActionTimestamps).forEach(sym => { if (!tokenBaselines[sym] && currentSymbols.has(sym)) { console.log(`🗑️ Clearing last action timestamp for ${sym} (no baseline).`); delete lastActionTimestamps[sym]; deletedKeys = true; } }); Object.keys(rebalanceState).forEach(sym => { if (!tokenBaselines[sym] && currentSymbols.has(sym)) { delete rebalanceState[sym]; /* No need to log, less noisy */ } }); Object.keys(adaptiveDeadZoneState).forEach(sym => { if (!tokenBaselines[sym] && currentSymbols.has(sym)) { delete adaptiveDeadZoneState[sym]; /* No need to log */ } }); if (deletedKeys) { stateChanged = true; } if (stateChanged) { saveState(); stateChanged = false; }

        // Calculate Portfolio Deviation (Original logic unchanged)
        let totalBaselineDifferenceManaged = 0; let totalManagedBaselineValue = 0; let managedAssetsCount = 0; portfolioSummary.forEach(row => { if (row.Baseline && typeof row.Baseline === 'number' && row.Baseline > 0 && !REBALANCE_EXCLUDE.includes(row.Symbol)) { totalBaselineDifferenceManaged += (row.Value - row.Baseline); totalManagedBaselineValue += row.Baseline; managedAssetsCount++; } }); let currentPortfolioDeviationPercent = 0; if (totalManagedBaselineValue > 0) { currentPortfolioDeviationPercent = (totalBaselineDifferenceManaged / totalManagedBaselineValue) * 100; }

        // Display Portfolio Table (Original logic unchanged)
        portfolioSummary.sort((a, b) => (b.Deviation || -Infinity) - (a.Deviation || -Infinity));
        if (portfolioSummary.length > 0) { const displayData = portfolioSummary.map(row => { return { Symbol: row.Symbol, Quantity: row.Quantity.toLocaleString(undefined, { maximumFractionDigits: 8 }), Price: row.Price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 10 }), Value: row.Value.toLocaleString(undefined, { style: 'currency', currency: 'USD' }), Baseline: row.Baseline ? `$${row.Baseline.toFixed(2)}` : 'N/A', Deviation: isNaN(row.Deviation) ? 'N/A' : `${(row.Deviation * 100).toFixed(2)}%` }; }); console.log("\n--- Portfolio Summary (Sorted by Deviation %) ---"); console.table(displayData); } else if (codes.length > 0) { console.log("ℹ️ Portfolio summary unavailable (missing prices?)."); }

        // Display Financial Overview (Original logic unchanged)
        console.log("--- Financial Overview ---"); console.log(`Total Holdings Value:   $${totalHoldingsValue.toFixed(2)}`); console.log(`Cash Balance:           $${cashBalance.toFixed(2)} ${QUOTE_CURRENCY}`); const totalPortfolioValue = totalHoldingsValue + cashBalance; console.log(`Total Portfolio Value:  $${totalPortfolioValue.toFixed(2)}`); const diffPrefixManaged = totalBaselineDifferenceManaged >= 0 ? '+' : ''; const diffColorManaged = totalBaselineDifferenceManaged >= 0 ? '\x1b[32m' : '\x1b[31m'; const resetColor = '\x1b[0m'; console.log(`Deviation (Managed):    ${diffColorManaged}${diffPrefixManaged}$${totalBaselineDifferenceManaged.toFixed(2)} (${currentPortfolioDeviationPercent.toFixed(2)}%)${resetColor} (${managedAssetsCount} Assets)`); console.log("--------------------------\n");

        // 6) Auto Trading Logic (Original strategy logic unchanged)
        const validPortfolioItems = portfolioSummary.filter(r => !isNaN(r.Deviation) && r.Baseline > 0);
        if (!api.isWsConnected() && requiredSymbols.length > 0) { console.warn("⚠️ WebSocket is disconnected. Price data may be stale or unavailable. Skipping trading actions this cycle."); }
        else if (!initialized) { console.log("⏳ Baselines initializing, skipping trading actions..."); }
        else if (codes.length > 0 && validPortfolioItems.length === 0) { console.log(`📉 No assets with valid data for decisions (Check WS/Prices for [${priceFetchIssues.join(', ')}]), skipping trading actions.`); }
        else if (codes.length === 0) { console.log("🧘 No holdings, skipping trading actions."); }
        else {
            console.log("🚦 Baselines ready. Proceeding with trading logic...");

            // --- Adaptive Dead Zone Activation/Deactivation Logic ---
            if (ENABLE_ADAPTIVE_DEAD_ZONE) {
                portfolioSummary.forEach(row => {
                    const sym = row.Symbol; const currentBaseline = tokenBaselines[sym];
                    if (!currentBaseline || currentBaseline <= 0 || HARVEST_EXCLUDE.includes(sym) || REBALANCE_EXCLUDE.includes(sym)) { if (adaptiveDeadZoneState[sym]) { delete adaptiveDeadZoneState[sym]; console.log(`ℹ️ ${sym}: Cleared adaptive DZ state (ineligible or excluded).`); } return; }
                    const deviation = row.Deviation; if (deviation === undefined || isNaN(deviation)) { if (adaptiveDeadZoneState[sym]) delete adaptiveDeadZoneState[sym]; return; }
                    const lastActionTime = lastActionTimestamps[sym] || 0; const timeSinceLastAction = Date.now() - lastActionTime; const inactivityTimeoutMet = timeSinceLastAction >= ADAPTIVE_DZ_INACTIVITY_TIMEOUT; const currentAdaptiveStatus = adaptiveDeadZoneState[sym] || false;
                    const isCurrentlyInOriginalDeadZone = deviation < FLAT_HARVEST_TRIGGER_PERCENT && deviation > -FLAT_REBALANCE_TRIGGER_PERCENT; // Strict check for activation
                    const isOnOrOutsideOriginalDeadZone = !isCurrentlyInOriginalDeadZone; // Use negation for deactivation check

                    // Deactivation: If currently adaptive AND ON or OUTSIDE the original boundaries
                    if (currentAdaptiveStatus && isOnOrOutsideOriginalDeadZone) { // Use the inverse for deactivation
                        delete adaptiveDeadZoneState[sym]; console.log(`✅ ${sym}: Adaptive DZ Mode DEACTIVATED (Deviation [${(deviation * 100).toFixed(2)}%] hit/exceeded original +/-${(FLAT_HARVEST_TRIGGER_PERCENT*100).toFixed(1)}/${(FLAT_REBALANCE_TRIGGER_PERCENT*100).toFixed(1)}% bounds).`);
                        if (trailingState[sym]) trailingState[sym].harvestCycleCount = 0; if (rebalanceState[sym]) rebalanceState[sym].rebalancePosCycleCount = 0;
                    }
                    // Activation: If NOT currently adaptive AND STRICTLY within original DZ AND timeout MET
                    else if (!currentAdaptiveStatus && isCurrentlyInOriginalDeadZone && inactivityTimeoutMet) { // Use strict check here
                        adaptiveDeadZoneState[sym] = true; console.log(`⚡ ${sym}: Adaptive DZ Mode ACTIVATED (In DZ & inactive for ${Math.round(timeSinceLastAction / (60*60*1000))} hrs). Using +/-${(ADAPTIVE_DZ_HARVEST_TRIGGER_PERCENT*100).toFixed(1)}% triggers.`);
                        if (trailingState[sym]) trailingState[sym].harvestCycleCount = 0; if (rebalanceState[sym]) rebalanceState[sym].rebalancePosCycleCount = 0;
                    }
                });
            } // --- End ADZ Logic ---

            // --- Revised Crash Protection Check ---
            if (ENABLE_CRASH_PROTECTION) { let assetsWithBaselineCount = 0; let assetsMeetingDeclineThresholdCount = 0; portfolioSummary.forEach(row => { if (row.Baseline && typeof row.Baseline === 'number' && row.Baseline > 0) { assetsWithBaselineCount++; if (row.Deviation !== undefined && !isNaN(row.Deviation) && row.Deviation <= CP_TRIGGER_MIN_NEGATIVE_DEV_PERCENT) { assetsMeetingDeclineThresholdCount++; } } }); if (assetsWithBaselineCount > 0) { const percentageMeetingThreshold = assetsMeetingDeclineThresholdCount / assetsWithBaselineCount; if (percentageMeetingThreshold >= CP_TRIGGER_ASSET_PERCENT) { isGlobalRiskSignalActive = true; console.log(`🛡️ Crash Protection ACTIVE (${(percentageMeetingThreshold * 100).toFixed(1)}% >= ${(CP_TRIGGER_ASSET_PERCENT * 100).toFixed(0)}% of assets <= ${(CP_TRIGGER_MIN_NEGATIVE_DEV_PERCENT * 100).toFixed(1)}% dev)`); } } }
            // --- End CP Check ---

            // --- Portfolio Override Harvest Logic ---
            let portfolioHarvestExecutedThisCycle = false;
            if (ENABLE_PORTFOLIO_HARVEST) {
                const portfolioHarvestTriggerValue = PORTFOLIO_HARVEST_TRIGGER_DEVIATION_PERCENT * 100;
                if (!portfolioHarvestState.flagged && currentPortfolioDeviationPercent >= portfolioHarvestTriggerValue) { portfolioHarvestState = { flagged: true, cycleCount: 0, flaggedAt: Date.now(), previousDeviationPercent: currentPortfolioDeviationPercent }; console.log(`📈 Portfolio flagged for Baseline Reset Harvest at ${currentPortfolioDeviationPercent.toFixed(2)}% (>= ${portfolioHarvestTriggerValue.toFixed(2)}%).`); }
                else if (portfolioHarvestState.flagged && currentPortfolioDeviationPercent < portfolioHarvestTriggerValue) { console.log(`📉 Portfolio dropped below Baseline Reset Harvest trigger. Clearing flag.`); portfolioHarvestState = { flagged: false, cycleCount: 0, previousDeviationPercent: null, flaggedAt: null }; }
                if (portfolioHarvestState.flagged) { const prevDev = portfolioHarvestState.previousDeviationPercent; if (prevDev !== null && typeof prevDev === 'number') { const currDev = currentPortfolioDeviationPercent; if (currDev < prevDev) { portfolioHarvestState.cycleCount++; console.log(`📊 P-Harvest: Dev decreased. Count INC to ${portfolioHarvestState.cycleCount}.`); } else if (currDev > prevDev) { portfolioHarvestState.cycleCount = Math.max(0, portfolioHarvestState.cycleCount - 1); console.log(`📊 P-Harvest: Dev increased. Count DEC to ${portfolioHarvestState.cycleCount}.`); } } portfolioHarvestState.previousDeviationPercent = currentPortfolioDeviationPercent; }
                if (portfolioHarvestState.flagged && portfolioHarvestState.cycleCount >= PORTFOLIO_HARVEST_CONFIRMATION_CYCLES) {
                    console.log(`🎉 Executing Portfolio Baseline Reset Harvest!`); portfolioHarvestExecutedThisCycle = true; let totalHarvestedThisEvent = 0; let assetsSoldCount = 0; const sellPromises = []; const assetsToUpdateTimestamp = [];
                    for (const row of validPortfolioItems) { if (HARVEST_EXCLUDE.includes(row.Symbol) || row.Value <= row.Baseline) continue; const originalBaseline = row.Baseline; const surplusUSD = row.Value - originalBaseline; if (surplusUSD < MIN_ASSET_SURPLUS_FOR_PORTFOLIO_HARVEST) continue; const qtyToSell = surplusUSD / row.Price; const qtyStr = api.roundQty(row.Symbol, qtyToSell);
                        if (parseFloat(qtyStr) > 0) { assetsSoldCount++; const sellPromise = (async () => { try { console.log(`   -> Selling P-Harvest surplus ${qtyStr} ${row.Symbol} (~$${surplusUSD.toFixed(2)})`); const sellResp = await api.placeSell(row.Symbol, qtyStr); if (sellResp?.id) { const effectiveSellPrice = getEffectivePriceFromResp(sellResp, row.Price); const actualSoldValue = parseFloat(qtyStr) * effectiveSellPrice; console.log(`   ✅ ${row.Symbol}: Sold ~$${actualSoldValue.toFixed(2)}. ID: ${sellResp.id}`); logTrade({ asset: row.Symbol, side: "SELL", quantity: qtyStr, price: effectiveSellPrice.toString(), clientOrderId: sellResp.id, note: `Portfolio Baseline Reset Harvest` }); tokenBaselines[row.Symbol] = originalBaseline; console.log(`   🔄 ${row.Symbol}: Baseline RESET to $${tokenBaselines[row.Symbol].toFixed(2)}.`); assetsToUpdateTimestamp.push(row.Symbol); if (trailingState[row.Symbol]) delete trailingState[row.Symbol]; return actualSoldValue; } else { console.warn(`   ⚠️ ${row.Symbol}: P-Harvest sell no ID. Baseline NOT reset.`); return 0; } } catch (err) { console.error(`   ❌ Error P-Harvest sell ${row.Symbol}:`, err.message); return 0; } })(); sellPromises.push(sellPromise); } }
                        const harvestedValues = await Promise.all(sellPromises); totalHarvestedThisEvent = harvestedValues.reduce((sum, val) => sum + val, 0);
                        if (assetsSoldCount > 0) { harvestedAmount += totalHarvestedThisEvent; anyTradesThisCycle = true; stateChanged = true; assetsToUpdateTimestamp.forEach(sym => { lastActionTimestamps[sym] = Date.now(); console.log(`   ⏱️ ${sym}: Updated last action timestamp (Portfolio Harvest).`); }); }
                        console.log(`🏁 P-Harvest finished. Total ~$${totalHarvestedThisEvent.toFixed(2)} from ${assetsSoldCount} assets.`); portfolioHarvestState = { flagged: false, cycleCount: 0, previousDeviationPercent: null, flaggedAt: null };
                }
            } // End Portfolio Harvest

            // --- Individual Asset Harvest Logic ---
            if (!portfolioHarvestExecutedThisCycle) {
                for (const row of validPortfolioItems) { const sym = row.Symbol; const currentBaseline = row.Baseline; if (HARVEST_EXCLUDE.includes(sym)) continue; const curP = row.currentPrice; const totalVal = row.usdValueNum; const currentDeviation = row.Deviation; const minOrderQty = assetMinOrderQuantities[sym] || 0; const minSellValue = minOrderQty > 0 ? minOrderQty * curP : 0; const isAdaptiveActive = ENABLE_ADAPTIVE_DEAD_ZONE && adaptiveDeadZoneState[sym]; const effectiveHarvestTriggerPercent = isAdaptiveActive ? ADAPTIVE_DZ_HARVEST_TRIGGER_PERCENT : FLAT_HARVEST_TRIGGER_PERCENT; const upperBandValue = currentBaseline * (1 + effectiveHarvestTriggerPercent);
                    if (!trailingState[sym]) { trailingState[sym] = { flagged: false, harvestCycleCount: 0, flaggedAt: null, previousDeviation: null }; } let st = trailingState[sym];
                    if (!st.flagged && totalVal >= upperBandValue) { st = { flagged: true, harvestCycleCount: 0, flaggedAt: Date.now(), previousDeviation: currentDeviation }; trailingState[sym] = st; let adaptiveNote = isAdaptiveActive ? ' (Adaptive)' : ''; console.log(`🚩 ${sym} flagged for Harvest${adaptiveNote} at $${totalVal.toFixed(2)} (Dev: ${(currentDeviation * 100).toFixed(2)}% >= ${(effectiveHarvestTriggerPercent*100).toFixed(2)}%)`); stateChanged = true; } else if (st.flagged && totalVal < upperBandValue) { let adaptiveNote = isAdaptiveActive ? ' (Adaptive)' : ''; console.log(`📉 ${sym} dropped below Harvest trigger${adaptiveNote} ($${upperBandValue.toFixed(2)}). Clearing flag.`); delete trailingState[sym]; stateChanged = true; continue; } if (!st.flagged) continue;
                    const flaggedDuration = Date.now() - (st.flaggedAt || Date.now());
                    if (flaggedDuration > FORCED_HARVEST_TIMEOUT) { const surplus = totalVal - currentBaseline; if (surplus < MIN_SURPLUS_FOR_FORCED_HARVEST || (minSellValue > 0 && surplus < minSellValue)) { const reason = surplus < MIN_SURPLUS_FOR_FORCED_HARVEST ? `Surplus $${surplus.toFixed(2)} < min $${MIN_SURPLUS_FOR_FORCED_HARVEST}` : `Surplus $${surplus.toFixed(2)} < min order value $${minSellValue.toFixed(2)}`; console.log(`ℹ️ ${sym} (Forced Harvest): ${reason}. Clearing flag.`); delete trailingState[sym]; stateChanged = true; continue; } const qtyToSell = surplus / curP; const qtyStr = api.roundQty(sym, qtyToSell);
                    if (parseFloat(qtyStr) > 0) { try { console.log(`⏳ Attempting Forced Harvest ${sym}: Selling ${qtyStr} (~$${surplus.toFixed(2)}) due to timeout.`); const resp = await api.placeSell(sym, qtyStr); if (resp?.id) { const effectiveSellPrice = getEffectivePriceFromResp(resp, curP); const sellValue = parseFloat(qtyStr) * effectiveSellPrice; console.log(`✅ (Forced Harvest) ${sym}: Sold ~$${sellValue.toFixed(2)}. ID: ${resp.id}`); logTrade({ asset: sym, side: "SELL", quantity: qtyStr, price: effectiveSellPrice.toString(), clientOrderId: resp.id, note: "Forced Harvest (Timeout)" }); harvestedAmount += sellValue; anyTradesThisCycle = true; let baselineAdjusted = false; let timestampUpdated = false; const wasAdaptiveActive = ENABLE_ADAPTIVE_DEAD_ZONE && adaptiveDeadZoneState[sym]; if (!wasAdaptiveActive) { tokenBaselines[sym] *= (1 + TARGET_ADJUST_PERCENT); console.log(`📈 ${sym}: Baseline increased to $${tokenBaselines[sym].toFixed(2)} (Std Adjust).`); baselineAdjusted = true; } else { console.log(`ℹ️ ${sym}: Baseline NOT adjusted (Adaptive Mode Active during Forced Harvest).`); } lastActionTimestamps[sym] = Date.now(); console.log(`⏱️ ${sym}: Updated last action timestamp (Forced Harvest).`); timestampUpdated = true; delete trailingState[sym]; if (baselineAdjusted || timestampUpdated) { stateChanged = true; } } else { console.warn(`⚠️ Forced Harvest ${sym}: sell order placed but no ID received. Clearing flag.`); delete trailingState[sym]; stateChanged = true; } } catch (err) { console.error(`❌ Error during Forced Harvest sell for ${sym}:`, err.message); } } else { console.log(`ℹ️ ${sym} (Forced Harvest): Rounded Qty '${qtyStr}' too small. Clearing flag.`); delete trailingState[sym]; stateChanged = true; } continue; }
                     if (st.previousDeviation !== null && typeof st.previousDeviation === 'number') { const prevDevPercent = (st.previousDeviation * 100).toFixed(2); const currDevPercent = (currentDeviation * 100).toFixed(2); if (currentDeviation < st.previousDeviation) { st.harvestCycleCount++; console.log(`📊 ${sym} Harvest: Dev decreased (${prevDevPercent}% -> ${currDevPercent}%). Count INC to ${st.harvestCycleCount}.`); stateChanged = true; } else if (currentDeviation > st.previousDeviation) { st.harvestCycleCount = Math.max(0, st.harvestCycleCount - 1); console.log(`📊 ${sym} Harvest: Dev increased (${prevDevPercent}% -> ${currDevPercent}%). Count DEC to ${st.harvestCycleCount}.`); stateChanged = true; } } else { st.harvestCycleCount = 0; } st.previousDeviation = currentDeviation;
                    const requiredHarvestCycles = isAdaptiveActive ? HARVEST_CYCLE_THRESHOLD + 1 : HARVEST_CYCLE_THRESHOLD;
                    if (st.harvestCycleCount >= requiredHarvestCycles) { const surplus = totalVal - currentBaseline; if (surplus < MIN_SURPLUS_FOR_HARVEST || (minSellValue > 0 && surplus < minSellValue)) { const reason = surplus < MIN_SURPLUS_FOR_HARVEST ? `Surplus $${surplus.toFixed(2)} < min $${MIN_SURPLUS_FOR_HARVEST}` : `Surplus $${surplus.toFixed(2)} < min order value $${minSellValue.toFixed(2)}`; console.log(`ℹ️ ${sym} (Harvest): ${reason}. Resetting count.`); st.harvestCycleCount = 0; stateChanged = true; continue; } const qtyToSell = surplus / curP; const qtyStr = api.roundQty(sym, qtyToSell);
                    if (parseFloat(qtyStr) > 0) { try { let adaptiveNote = isAdaptiveActive ? ` (Adaptive / ${requiredHarvestCycles} cycles)` : ` (${requiredHarvestCycles} cycles)`; console.log(`📉 Attempting Standard Harvest ${sym}: Selling ${qtyStr} (~$${surplus.toFixed(2)})${adaptiveNote}`); const resp = await api.placeSell(sym, qtyStr); if (resp?.id) { const effectiveSellPrice = getEffectivePriceFromResp(resp, curP); const sellValue = parseFloat(qtyStr) * effectiveSellPrice; console.log(`✅ Harvest ${sym}: Sold ~$${sellValue.toFixed(2)}. ID: ${resp.id}`); logTrade({ asset: sym, side: "SELL", quantity: qtyStr, price: effectiveSellPrice.toString(), clientOrderId: resp.id, note: `Harvest${adaptiveNote}` }); harvestedAmount += sellValue; anyTradesThisCycle = true; let baselineAdjusted = false; let timestampUpdated = false; const wasAdaptiveActive = ENABLE_ADAPTIVE_DEAD_ZONE && adaptiveDeadZoneState[sym]; if (!wasAdaptiveActive) { tokenBaselines[sym] *= (1 + TARGET_ADJUST_PERCENT); console.log(`📈 ${sym}: Baseline increased to $${tokenBaselines[sym].toFixed(2)} (Std Adjust).`); baselineAdjusted = true; } else { console.log(`ℹ️ ${sym}: Baseline NOT adjusted (Adaptive Mode Active).`); } lastActionTimestamps[sym] = Date.now(); console.log(`⏱️ ${sym}: Updated last action timestamp (Harvest).`); timestampUpdated = true; delete trailingState[sym]; if (baselineAdjusted || timestampUpdated) { stateChanged = true; } } else { console.warn(`⚠️ Harvest ${sym}: sell order placed but no ID received. Resetting count.`); st.harvestCycleCount = 0; st.previousDeviation = null; stateChanged = true; } } catch (err) { console.error(`❌ Error during Standard Harvest sell for ${sym}:`, err.message); } } else { console.log(`ℹ️ ${sym} (Harvest): Rounded Qty '${qtyStr}' too small. Resetting count.`); st.harvestCycleCount = 0; stateChanged = true; continue; } }
                }
            } // End Individual Harvest

            // --- Harvest Proceeds Allocation ---
            let amountForReinvest = 0; let amountForBTC = 0; let amountToCash = 0;
            // *** FIX V2.4.1-API-Merged-Fix2 START ***
            // Declare totalReinvestedThisCycle here so it's always defined in this scope
            let totalReinvestedThisCycle = 0;
            // *** FIX V2.4.1-API-Merged-Fix2 END ***

            if (harvestedAmount >= MIN_HARVEST_TO_ALLOCATE) {
                amountForReinvest = harvestedAmount * HARVEST_ALLOC_REINVEST_PERCENT;
                amountForBTC = harvestedAmount * HARVEST_ALLOC_BTC_PERCENT;
                console.log(`💵 Harvest Allocation: Total $${harvestedAmount.toFixed(2)} -> Reinvest: $${amountForReinvest.toFixed(2)}, BTC: $${amountForBTC.toFixed(2)}`);

                if (amountForReinvest > 0) {
                    let reinvestmentCandidates = portfolioSummary
                    .filter(row => !REBALANCE_EXCLUDE.includes(row.Symbol) && row.Baseline && row.Baseline > 0 && row.usdValueNum < row.Baseline && row.Deviation <= MIN_NEGATIVE_DEVIATION_FOR_REINVEST)
                    .sort((a, b) => a.Deviation - b.Deviation);

                    if (reinvestmentCandidates.length > 0) {
                        console.log(`💡 Found ${reinvestmentCandidates.length} candidate(s) for priority reinvestment (Dev <= ${MIN_NEGATIVE_DEVIATION_FOR_REINVEST * 100}%).`);
                        let remainingReinvestAllocation = amountForReinvest;
                        // totalReinvestedThisCycle = 0; // Initialization moved outside this block

                        for (const candidate of reinvestmentCandidates) {
                            if (remainingReinvestAllocation < MIN_REINVEST_BUY_USD) break;
                            const sym = candidate.Symbol;
                            const currentBaseline = candidate.Baseline;
                            const price = candidate.currentPrice;
                            const amountNeededToBaseline = Math.max(0, currentBaseline - candidate.usdValueNum);
                            if (amountNeededToBaseline <= 0.01) continue;
                            const buyAmountUSD = Math.min(amountNeededToBaseline, remainingReinvestAllocation);
                            if (buyAmountUSD < MIN_REINVEST_BUY_USD) continue;
                            const qtyToBuy = buyAmountUSD / price;
                            const minOrderQty = assetMinOrderQuantities[sym] || 0;
                            if (minOrderQty > 0 && qtyToBuy < minOrderQty) { continue; }
                            const qtyStr = api.roundQty(sym, qtyToBuy);
                            const qtyNum = parseFloat(qtyStr);

                            if (qtyNum > 0) {
                                try {
                                    console.log(`    R🛒 Attempting Priority Reinvestment ${sym}: Buying ${qtyStr} (~$${buyAmountUSD.toFixed(2)}) to reach baseline $${currentBaseline.toFixed(2)}.`);
                                    const resp = await api.placeBuy(sym, qtyStr);
                                    if (resp?.id) {
                                        const effectiveBuyPrice = getEffectivePriceFromResp(resp, price) || price;
                                        const actualCost = qtyNum * effectiveBuyPrice;
                                        console.log(`   ✅ Reinvest ${sym}: Spent ~$${actualCost.toFixed(2)}. ID: ${resp.id}`);
                                        logTrade({ asset: sym, side: "BUY", quantity: qtyStr, price: effectiveBuyPrice.toString(), clientOrderId: resp.id, note: `Priority Reinvestment Buy (from harvest)` });
                                        totalReinvestedThisCycle += actualCost;
                                        remainingReinvestAllocation -= actualCost;
                                        anyTradesThisCycle = true;
                                        lastActionTimestamps[sym] = Date.now();
                                        console.log(`   ⏱️ ${sym}: Updated last action timestamp (Priority Reinvest).`);
                                        console.log(`   ℹ️ ${sym}: Baseline remains $${tokenBaselines[sym].toFixed(2)} (Priority Reinvest - No Adjustment).`);
                                        if (rebalanceState[sym]) {
                                            console.log(`   🗑️ Clearing standard rebalance state for ${sym} after priority reinvestment.`);
                                            delete rebalanceState[sym];
                                        }
                                        stateChanged = true;
                                    } else {
                                        console.warn(`   ⚠️ Priority Reinvest ${sym}: Buy order placed but no ID received.`);
                                    }
                                } catch (err) {
                                    console.error(`   ❌ Error Priority Reinvest BUY ${sym}:`, err.message);
                                }
                            }
                        } // end for candidate loop
                        console.log(`🏁 Priority Reinvestment finished. Total spent: ~$${totalReinvestedThisCycle.toFixed(2)} / $${amountForReinvest.toFixed(2)} allocated.`);
                        amountToCash = (harvestedAmount - totalReinvestedThisCycle - amountForBTC); // Calculate cash AFTER potential reinvestment

                    } else {
                        console.log(`ℹ️ No candidates met priority reinvestment criteria.`);
                        amountToCash = harvestedAmount - amountForBTC; // No reinvestment occurred
                    }
                } else {
                    // No allocation for reinvestment
                    amountToCash = harvestedAmount - amountForBTC;
                }

                // Ensure cash allocation calculation is robust
                if (Math.abs((amountForBTC + totalReinvestedThisCycle + amountToCash) - harvestedAmount) > 0.01) {
                    console.warn(`WARN: Post-allocation check mismatch. Harvested: $${harvestedAmount.toFixed(2)}, BTC: $${amountForBTC.toFixed(2)}, Reinvested: $${totalReinvestedThisCycle.toFixed(2)}, Cash: $${amountToCash.toFixed(2)}`);
                    amountToCash = harvestedAmount - amountForBTC - totalReinvestedThisCycle; // Recalculate cash just in case
                }
                console.log(`   -> Final Allocation: Reinvested $${totalReinvestedThisCycle.toFixed(2)}, To BTC $${amountForBTC.toFixed(2)}, To Cash $${amountToCash.toFixed(2)}`);

            } else if (harvestedAmount > 0) {
                // Harvest occurred but was below allocation minimum
                amountToCash = harvestedAmount;
                amountForBTC = 0;
                amountForReinvest = 0;
                // totalReinvestedThisCycle is already 0
                console.log(`💵 Harvested $${harvestedAmount.toFixed(2)}, below minimum to allocate ($${MIN_HARVEST_TO_ALLOCATE}). Treating as cash.`);
            }


            // --- Auto BTC Buy from Harvest ---
            if (amountForBTC >= MIN_BTC_BUY_USD) { const btcBuyAmountUSD = amountForBTC; const btcSymbol = 'BTC'; const btcPrice = api.getLatestPrice(btcSymbol);
                if (btcPrice && btcPrice > 0) { const btcKrakenPair = api.getKrakenPairName(btcSymbol);
                    if (btcKrakenPair) { const btcQty = btcBuyAmountUSD / btcPrice; const qtyStr = api.roundQty(btcSymbol, btcQty); const qtyNumBTC = parseFloat(qtyStr);
                        if (qtyNumBTC > 0) { let currentCash = cashBalance; try { [currentCash] = await api.getHoldings(); } catch (err) { console.warn("Warn: Failed to re-fetch balance before BTC buy."); }
                        if (currentCash >= btcBuyAmountUSD) { try { console.log(`₿ Attempting Auto BTC Buy: ${qtyStr} (~$${btcBuyAmountUSD.toFixed(2)}) [Allocated: ${HARVEST_ALLOC_BTC_PERCENT*100}%]`); const buyResp = await api.placeBuy(btcSymbol, qtyStr); if (buyResp?.id) { const effectiveBuyPriceBTC = getEffectivePriceFromResp(buyResp, btcPrice); const actualCost = qtyNumBTC * effectiveBuyPriceBTC; console.log(`✅ BTC Buy ~$${actualCost.toFixed(2)}. ID: ${buyResp.id}`); logTrade({ asset: btcSymbol, side: "BUY", quantity: qtyStr, price: effectiveBuyPriceBTC.toString(), clientOrderId: buyResp.id, note: `Auto BTC Buy (Allocated from Harvest)` }); anyTradesThisCycle = true; const existingQtyBTC = portfolioSummary.find(r => r.Symbol === 'BTC')?.rawQuantity || 0; const newTotalQtyBTC = existingQtyBTC + qtyNumBTC; const newTotalValueBTC = newTotalQtyBTC * effectiveBuyPriceBTC; if (newTotalValueBTC > 0.01) { tokenBaselines['BTC'] = newTotalValueBTC; console.log(`🔄 BTC: Baseline RESET to $${tokenBaselines['BTC'].toFixed(2)}.`); } lastActionTimestamps['BTC'] = Date.now(); console.log(`⏱️ BTC: Updated last action timestamp (Auto Buy).`); stateChanged = true; } else { console.warn(`⚠️ BTC Buy: order placed but no ID received.`); } } catch (err) { console.error("❌ Error during BTC auto-buy:", err.message); } } else { console.warn(`💰 Insufficient cash ($${currentCash.toFixed(2)}) for auto BTC buy ($${btcBuyAmountUSD.toFixed(2)})`); } } } else { console.warn(`⚠️ Cannot Auto Buy BTC: No valid Kraken pair found for BTC.`); } } else { console.warn(`⚠️ Cannot Auto Buy BTC: Price unavailable via WebSocket.`); } } else if (harvestedAmount >= MIN_HARVEST_TO_ALLOCATE && HARVEST_ALLOC_BTC_PERCENT > 0 && amountForBTC > 0) { console.log(`ℹ️ BTC allocation $${amountForBTC.toFixed(2)} is less than minimum buy $${MIN_BTC_BUY_USD}. Skipping BTC buy.`); }

                        // --- Rebalancing Logic (Standard) ---
                        for (const row of validPortfolioItems) { const sym = row.Symbol; const currentBaseline = row.Baseline; if (REBALANCE_EXCLUDE.includes(sym) || trailingState[sym]?.flagged) { if (rebalanceState[sym]) { delete rebalanceState[sym]; } continue; } const curP = row.currentPrice; const totalVal = row.usdValueNum; const currentDeviation = row.Deviation; const minOrderQty = assetMinOrderQuantities[sym] || 0; const minBuyValue = minOrderQty > 0 ? minOrderQty * curP : 0; const isAdaptiveActive = ENABLE_ADAPTIVE_DEAD_ZONE && adaptiveDeadZoneState[sym]; const effectiveRebalanceTriggerPercent = isAdaptiveActive ? ADAPTIVE_DZ_REBALANCE_TRIGGER_PERCENT : FLAT_REBALANCE_TRIGGER_PERCENT; const lowerBandValue = currentBaseline * (1 - effectiveRebalanceTriggerPercent);
                        if (totalVal >= lowerBandValue) { if (rebalanceState[sym]) { console.log(`📈 ${sym}: Value recovered above rebalance trigger ($${lowerBandValue.toFixed(2)}). Clearing rebalance state.`); delete rebalanceState[sym]; } continue; }
                        if (!rebalanceState[sym]) { let adaptiveNote = isAdaptiveActive ? ' (Adaptive)' : ''; rebalanceState[sym] = { triggered: true, triggeredAt: Date.now(), rebalancePosCycleCount: 0, attemptCount: 0, cooldownUntil: 0, currentBaselineWhenTriggered: currentBaseline, previousDeviation: currentDeviation }; console.log(`⚖️ ${sym}: Rebalance triggered${adaptiveNote} at $${totalVal.toFixed(2)} (Dev: ${(currentDeviation * 100).toFixed(2)}% <= -${(effectiveRebalanceTriggerPercent*100).toFixed(2)}%)`); } let rSt = rebalanceState[sym];
                        const rebalanceActiveDuration = Date.now() - (rSt.triggeredAt || Date.now());
                        if (rebalanceActiveDuration > FORCE_REBALANCE_TIMEOUT) { const shortfall = rSt.currentBaselineWhenTriggered - totalVal; const desiredForcedAmountUSD = shortfall * FORCE_REBALANCE_SHORTFALL_PERCENT; if (desiredForcedAmountUSD < MIN_FORCED_REBALANCE_USD) { console.log(`ℹ️ ${sym} (Forced Rebalance): Calc amount $${desiredForcedAmountUSD.toFixed(2)} < min $${MIN_FORCED_REBALANCE_USD}. Clearing state.`); delete rebalanceState[sym]; continue; } let qtyToActuallyBuy = desiredForcedAmountUSD / curP; let costOfTrade = desiredForcedAmountUSD; let noteSuffix = ` (Timeout)`; if (minOrderQty > 0 && (desiredForcedAmountUSD / curP) < minOrderQty) { if (minBuyValue >= MIN_FORCED_REBALANCE_USD && cashBalance >= minBuyValue) { qtyToActuallyBuy = minOrderQty; costOfTrade = minBuyValue; noteSuffix += " (Min Override)"; console.log(`ℹ️ ${sym} (Forced Rebalance): Overriding to min qty ${qtyToActuallyBuy} (~$${costOfTrade.toFixed(2)}).`); } else { console.log(`ℹ️ ${sym} (Forced Rebalance): Cannot override (Min Val ~$${minBuyValue.toFixed(2)} vs Cash $${cashBalance.toFixed(2)}). Clearing state.`); delete rebalanceState[sym]; continue; } }
                        let currentCash = cashBalance; try { [currentCash] = await api.getHoldings(); } catch (err) { console.warn("Warn: Failed to re-fetch balance before Forced Rebalance."); }
                        if (currentCash >= costOfTrade) { const qtyStr = api.roundQty(sym, qtyToActuallyBuy); if (parseFloat(qtyStr) > 0) { try { console.log(`⏳ Forced Rebalance${noteSuffix} ${sym}: Buying ${qtyStr} (~$${costOfTrade.toFixed(2)})`); const resp = await api.placeBuy(sym, qtyStr); if (resp?.id) { const effectiveBuyPrice = getEffectivePriceFromResp(resp, curP); const actualCost = parseFloat(qtyStr) * effectiveBuyPrice; console.log(`✅ Forced Rebalance ${sym}: Bought ~$${actualCost.toFixed(2)}. ID: ${resp.id}`); logTrade({ asset: sym, side: "BUY", quantity: qtyStr, price: effectiveBuyPrice.toString(), clientOrderId: resp.id, note: `Forced Rebalance BUY${noteSuffix}` }); anyTradesThisCycle = true; let baselineAdjusted = false; let timestampUpdated = false; const wasAdaptiveActive = ENABLE_ADAPTIVE_DEAD_ZONE && adaptiveDeadZoneState[sym]; if (!wasAdaptiveActive) { tokenBaselines[sym] *= (1 - TARGET_ADJUST_PERCENT); console.log(`📉 ${sym}: Baseline decreased to $${tokenBaselines[sym].toFixed(2)} (Std Adjust).`); baselineAdjusted = true; } else { console.log(`ℹ️ ${sym}: Baseline NOT adjusted (Adaptive Mode Active during Forced Rebalance).`); } lastActionTimestamps[sym] = Date.now(); console.log(`⏱️ ${sym}: Updated last action timestamp (Forced Rebalance).`); timestampUpdated = true; delete rebalanceState[sym]; if (baselineAdjusted || timestampUpdated) { stateChanged = true; } continue; } else { console.warn(`⚠️ Forced Rebalance ${sym}: buy order placed but no ID received. Clearing state.`); delete rebalanceState[sym]; continue; } } catch (err) { console.error(`❌ Error Forced Rebalance BUY ${sym}:`, err.message); } } else { console.log(`ℹ️ ${sym} (Forced Rebalance): Rounded Qty '${qtyStr}' too small. Clearing state.`); delete rebalanceState[sym]; continue; } } else { console.log(`💰 Insufficient cash ($${currentCash.toFixed(2)}) for forced rebalance ${sym} ($${costOfTrade.toFixed(2)}).`); } }
                        if (Date.now() < rSt.cooldownUntil) continue;
                        if (rSt.previousDeviation !== null && typeof rSt.previousDeviation === 'number') { const prevDevPercent = (rSt.previousDeviation * 100).toFixed(2); const currDevPercent = (currentDeviation * 100).toFixed(2); if (currentDeviation > rSt.previousDeviation) { rSt.rebalancePosCycleCount++; console.log(`📊 ${sym} rebalance: Dev recovered (${prevDevPercent}% -> ${currDevPercent}%). Count INC to ${rSt.rebalancePosCycleCount}.`); } else if (currentDeviation < rSt.previousDeviation) { rSt.rebalancePosCycleCount = Math.max(0, rSt.rebalancePosCycleCount - 1); console.log(`📊 ${sym} rebalance: Dev dropped (${prevDevPercent}% -> ${currDevPercent}%). Count DEC to ${rSt.rebalancePosCycleCount}.`); } } else { rSt.rebalancePosCycleCount = 0; } rSt.previousDeviation = currentDeviation;
                        const baseEffectiveRebalanceThreshold = REBALANCE_POSITIVE_THRESHOLD + (isGlobalRiskSignalActive ? CRASH_PROTECTION_THRESHOLD_INCREASE : 0); const requiredRebalanceCycles = isAdaptiveActive ? baseEffectiveRebalanceThreshold + 1 : baseEffectiveRebalanceThreshold; const effectivePartialRecoveryPercent = isGlobalRiskSignalActive ? (PARTIAL_RECOVERY_PERCENT * (CRASH_PROTECTION_PARTIAL_RECOVERY_PERCENT / 0.875)) /* Adjusted CP Recovery % */ : PARTIAL_RECOVERY_PERCENT;
                        if (rSt.rebalancePosCycleCount >= requiredRebalanceCycles) { const shortfall = rSt.currentBaselineWhenTriggered - totalVal; const desiredPartialAmountUSD = shortfall * effectivePartialRecoveryPercent; let logNote = isGlobalRiskSignalActive ? ` (CP Active)` : ``; logNote += isAdaptiveActive ? ` (Adaptive / ${requiredRebalanceCycles} cycles)` : ` (Std / ${requiredRebalanceCycles} cycles)`; logNote += ` (Rec ${effectivePartialRecoveryPercent.toFixed(3)})`; if (desiredPartialAmountUSD < MIN_PARTIAL_REBALANCE_USD) { console.log(`ℹ️ ${sym} (Std Rebalance): Amount $${desiredPartialAmountUSD.toFixed(2)} < min $${MIN_PARTIAL_REBALANCE_USD}${logNote}. Resetting count.`); rSt.rebalancePosCycleCount = 0; rSt.previousDeviation = null; continue; } let qtyToActuallyBuy = desiredPartialAmountUSD / curP; let costOfTrade = desiredPartialAmountUSD; let noteSuffix = ``; if (minOrderQty > 0 && (desiredPartialAmountUSD / curP) < minOrderQty) { if (minBuyValue >= MIN_PARTIAL_REBALANCE_USD && cashBalance >= minBuyValue) { qtyToActuallyBuy = minOrderQty; costOfTrade = minBuyValue; noteSuffix = " (Min Override)"; console.log(`ℹ️ ${sym} (Std Rebalance): Overriding to min qty ${qtyToActuallyBuy} (~$${costOfTrade.toFixed(2)})${logNote}.`); } else { console.log(`ℹ️ ${sym} (Std Rebalance): Cannot override (Min Val ~$${minBuyValue.toFixed(2)} vs Cash $${cashBalance.toFixed(2)})${logNote}. Resetting count.`); rSt.rebalancePosCycleCount = 0; rSt.previousDeviation = null; continue; } }
                        let currentCash = cashBalance; try { [currentCash] = await api.getHoldings(); } catch (err) { console.warn("Warn: Failed to re-fetch balance before Standard Rebalance."); }
                        if (currentCash >= costOfTrade) { const qtyStr = api.roundQty(sym, qtyToActuallyBuy); if (parseFloat(qtyStr) > 0) { try { console.log(`📈 Standard Rebalance BUY${noteSuffix} ${sym}: ${qtyStr} (~$${costOfTrade.toFixed(2)})${logNote}.`); const resp = await api.placeBuy(sym, qtyStr); if (resp?.id) { const effectiveBuyPrice = getEffectivePriceFromResp(resp, curP); const actualCost = parseFloat(qtyStr) * effectiveBuyPrice; console.log(`✅ Rebalance BUY ${sym}: Spent ~$${actualCost.toFixed(2)}. ID: ${resp.id}`); const tradeNote = `Rebalance BUY (Attempt ${rSt.attemptCount + 1})${noteSuffix}${logNote}`; logTrade({ asset: sym, side: "BUY", quantity: qtyStr, price: effectiveBuyPrice.toString(), clientOrderId: resp.id, note: tradeNote }); anyTradesThisCycle = true; let baselineAdjusted = false; let timestampUpdated = false; const wasAdaptiveActive = ENABLE_ADAPTIVE_DEAD_ZONE && adaptiveDeadZoneState[sym]; if (!wasAdaptiveActive) { tokenBaselines[sym] *= (1 - TARGET_ADJUST_PERCENT); console.log(`📉 ${sym}: Baseline decreased to $${tokenBaselines[sym].toFixed(2)} (Std Adjust).`); baselineAdjusted = true; } else { console.log(`ℹ️ ${sym}: Baseline NOT adjusted (Adaptive Mode Active).`); } lastActionTimestamps[sym] = Date.now(); console.log(`⏱️ ${sym}: Updated last action timestamp (Rebalance).`); timestampUpdated = true; if (baselineAdjusted || timestampUpdated) { stateChanged = true; } rSt.attemptCount++; rSt.rebalancePosCycleCount = 0; rSt.previousDeviation = null; const newValueAfterBuy = totalVal + actualCost; const originalLowerBandValue = rSt.currentBaselineWhenTriggered * (1 - effectiveRebalanceTriggerPercent); if (newValueAfterBuy >= originalLowerBandValue) { console.log(`👍 ${sym}: Value $${newValueAfterBuy.toFixed(2)} recovered above trigger band $${originalLowerBandValue.toFixed(2)}. Clearing state.`); delete rebalanceState[sym]; continue; } if (rSt.attemptCount >= MAX_REBALANCE_ATTEMPTS) { rSt.cooldownUntil = Date.now() + REBALANCE_COOLDOWN; console.log(`⏸️ ${sym}: Max rebalance attempts (${MAX_REBALANCE_ATTEMPTS}) reached. Cooldown until ${new Date(rSt.cooldownUntil).toLocaleTimeString()}.`); rSt.attemptCount = 0; } } else { console.warn(`⚠️ Rebalance BUY ${sym}: order placed but no ID received. Resetting count.`); rSt.rebalancePosCycleCount = 0; rSt.previousDeviation = null; } } catch (err) { console.error(`❌ Error Standard Rebalance BUY ${sym}:`, err.message); } } else { console.log(`ℹ️ ${sym} (Std Rebalance): Rounded Qty '${qtyStr}' too small${logNote}. Resetting count.`); rSt.rebalancePosCycleCount = 0; rSt.previousDeviation = null; } } else { console.log(`💰 Insufficient cash ($${currentCash.toFixed(2)}) for rebalance ${sym} ($${costOfTrade.toFixed(2)})${logNote}. Resetting count.`); rSt.rebalancePosCycleCount = 0; rSt.previousDeviation = null; } }
                        } // End Rebalance Loop

                        // Final state save if anything changed during trading logic
                        if (stateChanged) { saveState(); }

        } // --- End Auto Trading Logic Block ---

        // --- Display Active States (Original logic unchanged) ---
        try { if (portfolioHarvestState.flagged) { console.log(`📈 Portfolio Harvest Flagged: Count ${portfolioHarvestState.cycleCount}/${PORTFOLIO_HARVEST_CONFIRMATION_CYCLES}, Prev Port. Dev: ${portfolioHarvestState.previousDeviationPercent?.toFixed(2)}%`); } const flaggedHarvest = Object.entries(trailingState).filter(([sym, s]) => s?.flagged && tokenBaselines[sym]).map(([sym, s]) => { const isAdaptive = ENABLE_ADAPTIVE_DEAD_ZONE && adaptiveDeadZoneState[sym]; const reqCycles = isAdaptive ? HARVEST_CYCLE_THRESHOLD + 1 : HARVEST_CYCLE_THRESHOLD; return `${sym}(${isAdaptive ? 'A:' : ''}${s.harvestCycleCount}/${reqCycles})`; }); if (flaggedHarvest.length > 0) console.log(`🚩 Flagged Harvest: [${flaggedHarvest.join(", ")}]`); const activeRebalancing = Object.entries(rebalanceState).filter(([sym, s]) => s?.triggered && Date.now() >= (s.cooldownUntil || 0) && tokenBaselines[sym]).map(([sym, s]) => { const isAdaptive = ENABLE_ADAPTIVE_DEAD_ZONE && adaptiveDeadZoneState[sym]; const baseThresh = REBALANCE_POSITIVE_THRESHOLD + (isGlobalRiskSignalActive ? CRASH_PROTECTION_THRESHOLD_INCREASE : 0); const reqCycles = isAdaptive ? baseThresh + 1 : baseThresh; const attemptInfo = s.attemptCount > 0 ? `/A:${s.attemptCount}` : ''; return `${sym}(${isAdaptive ? 'A:' : ''}${s.rebalancePosCycleCount}/${reqCycles}${attemptInfo})`; }); const inCooldown = Object.entries(rebalanceState).filter(([sym, s]) => s?.triggered && Date.now() < (s.cooldownUntil || 0) && tokenBaselines[sym]).map(([sym, s]) => `${sym}(CD:${Math.ceil(((s.cooldownUntil || 0) - Date.now()) / 60000)}m)`); if (activeRebalancing.length > 0) console.log(`⚖️ Active Rebalance: [${activeRebalancing.join(", ")}]`); if (inCooldown.length > 0) console.log(`⏸️ Rebalance Cooldown: [${inCooldown.join(", ")}]`); const adaptiveAssets = Object.entries(adaptiveDeadZoneState).filter(([sym, isActive]) => isActive && tokenBaselines[sym]).map(([sym]) => sym); if (adaptiveAssets.length > 0) { console.log(`⚡ Adaptive DZ Active: [${adaptiveAssets.join(", ")}]`); } if (!anyTradesThisCycle && !portfolioHarvestState.flagged && flaggedHarvest.length === 0 && activeRebalancing.length === 0 && inCooldown.length === 0 && adaptiveAssets.length === 0 && initialized && validPortfolioItems.length > 0) { console.log("🧘 No trading actions or adaptive states triggered this cycle."); } else if (!initialized) { /* Logged earlier */ } else if (!anyTradesThisCycle && validPortfolioItems.length === 0 && codes.length > 0) { console.log("🧘 No action (waiting for valid data)."); } }
        catch (displayError) { console.error("⚠️ Error displaying states:", displayError); }

        // --- Cycle Timing ---
        const endTime = Date.now(); const elapsed = endTime - startTime;
        const delay = Math.max(0, REFRESH_INTERVAL - elapsed);
        console.log(`----- Cycle End: Took ${elapsed}ms. Waiting ${delay}ms... -----`);
        await wait(delay);

    } // End while(true) loop

    console.log("🛑 Main loop exited unexpectedly.");
    if(api) api.close();
    rl.close();
}

// --- Graceful Exit & Exception Handling ---
function gracefulShutdown(signal) { console.log(`\n🚦 Received ${signal}. Shutting down...`); console.log("💾 Saving final state..."); saveState(); if (api) { console.log("🔌 Closing API connections..."); api.close(); } rl.close(); console.log("✅ Shutdown complete."); process.exit(0); }
const signals = ["SIGINT", "SIGTERM", "SIGQUIT"]; signals.forEach((signal) => { process.removeAllListeners(signal); process.on(signal, () => gracefulShutdown(signal)); });
process.on('unhandledRejection', (reason, promise) => { console.error('🚨 Unhandled Rejection:', reason); });
process.on('uncaughtException', (err, origin) => { console.error('💥 Uncaught Exception:', err, 'Origin:', origin); gracefulShutdown('uncaughtException'); setTimeout(() => process.exit(1), 2000).unref(); });

// --- Main Execution ---
console.log(`🏁 Starting Kraken Cryptobot Script (v2.4.1-API-Merged-Fix2)...`);
const { loadedBaselines, loadedTrailingState, loadedLastActionTimestamps } = loadState();
tokenBaselines = loadedBaselines; trailingState = loadedTrailingState; lastActionTimestamps = loadedLastActionTimestamps;

mainLoop().catch((err) => {
    console.error("💥 FATAL ERROR in main execution scope:", err); if (err.stack) { console.error(err.stack); }
    console.log("💾 Attempting to save state on error..."); saveState();
    if (api) { try { api.close(); } catch (closeErr) { console.error("Error closing API during shutdown:", closeErr);} }
    rl.close();
    process.exit(1);
});

// ==================== Change Log ====================
// v2.4.1-API-Merged-Fix2: Fixed ReferenceError for totalReinvestedThisCycle scope.
// v2.4.1-API-Merged-Fix1: Prevent initial WS price check from resetting valid loaded baselines.
// v2.4.1-API-Merged: (As described previously)
// ...
// ======================================================

// ----- Required Dependencies -----
// npm install dotenv ws node-fetch@^2 chalk sparkly console [--save]
// yarn add dotenv ws node-fetch@^2 chalk sparkly console
// -------------------------------