// marketEngine.js - Market simulation engine for the bot arena
const pricingModels = require('./pricingModels');

class MarketEngine {
  constructor() {
    // Market & Contract Specs (from the original HTML)
    this.INITIAL_PRICE = 50000;        // BTC-USD
    this.VOLATILITY = 0.45;           // 45% annual
    this.RISK_FREE = 0.03;            // 3% annual
    this.STRIKE = this.INITIAL_PRICE;
    this.DAYS_TO_EXP = 7;
    this.SECONDS_PER_DAY = 24 * 60 * 60;
    this.secondsPerYear = this.SECONDS_PER_DAY * 365.25;

    // Market state
    this.price = this.INITIAL_PRICE;
    this.history = [this.INITIAL_PRICE];
    this.T = this.DAYS_TO_EXP / 365.25; // Time to expiration in years
    this.startTime = Date.now();
    this.tick = 0;
    
    // Simulation state
    this.isRunning = false;
    this.simulationInterval = null;
    this.speed = 500; // ms between steps (matches the HTML default)
  }

  // Gaussian random number generator (Box-Muller transform)
  randn() {
    let u = 0, v = 0;
    while(u === 0) u = Math.random(); // Converting [0,1) to (0,1)
    while(v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  }

  // Calculate 24h realized volatility based on price history
  volatility() {
    if (this.history.length < 20) return this.VOLATILITY;
    
    const returns = [];
    for (let i = 1; i < 20 && i < this.history.length; i++) {
      returns.push(Math.log(this.history[this.history.length - 20 + i] / 
                           this.history[this.history.length - 20 + i - 1]));
    }
    
    const mean = returns.reduce((a, b) => a + b) / returns.length;
    const varr = returns.map(r => Math.pow(r - mean, 2))
                       .reduce((a, b) => a + b) / (returns.length - 1);
    return Math.sqrt(varr * this.SECONDS_PER_DAY * 365.25); // annualized
  }

  // Single simulation step
  step() {
    // 1. Evolve price (geometric brownian motion)
    const dt = 1 / this.SECONDS_PER_DAY;
    const drift = this.RISK_FREE - 0.5 * this.VOLATILITY * this.VOLATILITY;
    const Z = this.randn();
    this.price *= Math.exp(drift * dt + this.VOLATILITY * Math.sqrt(dt) * Z);
    this.history.push(this.price);
    
    // Keep history to last 200 points
    if (this.history.length > 200) {
      this.history.shift();
    }

    // 2. Time decay
    const elapsedSeconds = (Date.now() - this.startTime) / 1000;
    this.T = Math.max(0, (this.DAYS_TO_EXP * this.SECONDS_PER_DAY - elapsedSeconds) / this.secondsPerYear);

    // 3. Check for expiry
    if (this.T <= 0) {
      this.handleExpiry();
    }

    this.tick++;
  }

  // Handle option expiry
  handleExpiry() {
    console.log(`Options expired. Settlement price: ${this.price.toFixed(2)}, Strike: ${this.STRIKE}`);
    
    // Reset expiration time
    this.T = this.DAYS_TO_EXP / 365.25;
    this.startTime = Date.now();
  }

  // Start the simulation
  start(speed = 500) {
    if (this.isRunning) {
      console.log('Simulation already running');
      return;
    }

    this.speed = speed;
    this.isRunning = true;
    
    this.simulationInterval = setInterval(() => {
      this.step();
    }, this.speed);
    
    console.log(`Simulation started with speed: ${speed}ms`);
  }

  // Stop the simulation
  stop() {
    if (!this.isRunning) {
      console.log('Simulation not running');
      return;
    }

    clearInterval(this.simulationInterval);
    this.isRunning = false;
    this.simulationInterval = null;
    
    console.log('Simulation stopped');
  }

  // Reset the simulation to initial state
  reset() {
    this.stop();
    
    this.price = this.INITIAL_PRICE;
    this.history = [this.INITIAL_PRICE];
    this.tick = 0;
    this.T = this.DAYS_TO_EXP / 365.25;
    this.startTime = Date.now();
    
    console.log('Simulation reset to initial state');
  }

  // Get current market state for API/clients
  getState() {
    return {
      price: this.price,
      history: this.history.slice(-20), // Last 20 prices for display
      T: this.T,
      volatility: this.volatility(),
      isRunning: this.isRunning,
      tick: this.tick,
      startTime: this.startTime
    };
  }
}

module.exports = MarketEngine;