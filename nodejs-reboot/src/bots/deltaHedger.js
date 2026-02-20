// deltaHedger.js - Delta hedging strategy bot
const BaseBot = require('./baseBot');
const pricingModels = require('../market/pricingModels');

class DeltaHedger extends BaseBot {
  constructor(name, color, cash = 1e6) {
    super(name, color, cash);
  }

  update(market) {
    // Calculate time to expiration (with minimum value to avoid division by zero)
    const T = Math.max(market.T, 1 / market.secondsPerYear);
    
    // Calculate delta of the options position we're hedging
    const d = pricingModels.delta(market.price, this.STRIKE, T, this.VOLATILITY, this.RISK_FREE, true);
    
    // Target number of shares to maintain a delta-neutral position
    const targetShares = -this.options * d;
    
    // Calculate how many shares we need to trade to reach the target
    const deltaShares = targetShares - this.shares;
    
    // If the adjustment is too small, skip the trade
    if (Math.abs(deltaShares) < 0.01) return;
    
    if (deltaShares > 0) { // Need to buy shares to hedge
      const qty = deltaShares;
      const cost = qty * market.price;
      
      if (cost <= this.cash) {
        this.cash -= cost;
        this.shares += qty;
        this.logMsg(`HEDGE BUY ${qty.toFixed(2)} @ ${market.price.toFixed(0)}`);
      }
    } else { // Need to sell shares to hedge
      const qty = -deltaShares;
      this.cash += qty * market.price;
      this.shares -= qty;
      this.logMsg(`HEDGE SELL ${qty.toFixed(2)} @ ${market.price.toFixed(0)}`);
    }
  }
}

module.exports = DeltaHedger;