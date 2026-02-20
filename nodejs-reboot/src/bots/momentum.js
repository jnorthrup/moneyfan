// momentum.js - Momentum strategy bot
const BaseBot = require('./baseBot');
const pricingModels = require('../market/pricingModels');

class Momentum extends BaseBot {
  constructor(name, color, cash = 1e6) {
    super(name, color, cash);
  }

  update(market) {
    // Calculate return over the last 20 price points
    if (market.history.length < 20) return; // Not enough data yet
    
    const startPrice = market.history[Math.max(0, market.history.length - 20)];
    const ret = (market.price - startPrice) / startPrice;
    
    // Buy call options if there's strong positive momentum
    if (ret > 0.02 && this.cash > 5000) {
      const qty = 1;
      const T = Math.max(market.T, 1 / market.secondsPerYear);
      const optPrice = pricingModels.blackScholes(
        market.price, 
        this.STRIKE, 
        T, 
        this.VOLATILITY, 
        this.RISK_FREE, 
        true
      );
      
      if (this.cash >= optPrice) {
        this.options += qty;
        this.cash -= optPrice;
        this.logMsg(`MOMO CALL ${qty} @ ${optPrice.toFixed(0)}`);
      }
    }
    
    // Sell options if there's strong negative momentum and we have options to sell
    if (ret < -0.02 && this.options > 1) {
      const qty = 1;
      const T = Math.max(market.T, 1 / market.secondsPerYear);
      const optPrice = pricingModels.blackScholes(
        market.price, 
        this.STRIKE, 
        T, 
        this.VOLATILITY, 
        this.RISK_FREE, 
        true
      );
      
      this.options -= qty;
      this.cash += optPrice;
      this.logMsg(`MOMO SELL ${qty} @ ${optPrice.toFixed(0)}`);
    }
  }
}

module.exports = Momentum;