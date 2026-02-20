// baseBot.js - Base class for all trading bots
const pricingModels = require('../market/pricingModels');

class BaseBot {
  constructor(name, color, cash = 1e6) {
    this.name = name;
    this.color = color;
    this.cash = cash;
    this.shares = 0;
    this.options = 0;          // net position (+=long,-=short)
    this.premium = 0;          // total premium paid/received
    this.log = [];
    this.STRIKE = 50000;       // Match the market's strike price
    this.VOLATILITY = 0.45;    // Match the market's volatility
    this.RISK_FREE = 0.03;     // Match the market's risk-free rate
  }

  logMsg(msg) {
    this.log.push({
      time: new Date().toISOString(),
      message: msg
    });
    
    // Keep only the last 12 log entries
    if (this.log.length > 12) {
      this.log.shift();
    }
  }

  // Calculate the current value of the bot's portfolio
  value(marketPrice) {
    const optionValue = this.options > 0 
      ? this.options * Math.max(marketPrice - this.STRIKE, 0) 
      : this.options * Math.min(marketPrice - this.STRIKE, 0);
    return this.cash + this.shares * marketPrice + optionValue;
  }

  // Update state based on market movement
  update(market) {
    // This method should be overridden by specific bot strategies
    // to implement their trading logic
  }

  // Get bot state for API/clients
  getState(marketPrice) {
    return {
      name: this.name,
      color: this.color,
      cash: this.cash,
      shares: this.shares,
      options: this.options,
      pnl: this.value(marketPrice) - 1e6, // Profit/Loss since start
      log: this.log,
      value: this.value(marketPrice)
    };
  }
}

module.exports = BaseBot;