// volArb.js - Volatility arbitrage strategy bot
const BaseBot = require('./baseBot');
const pricingModels = require('../market/pricingModels');

class VolArb extends BaseBot {
  constructor(name, color, cash = 1e6) {
    super(name, color, cash);
    this.targetVol = 0.45; // Target volatility to trade against
  }

  update(market) {
    // Calculate realized volatility from market
    const realized = market.volatility();
    
    // Calculate time to expiration (with minimum value to avoid division by zero)
    const T = Math.max(market.T, 1 / market.secondsPerYear);
    
    // Calculate option price using our target volatility
    const optPrice = pricingModels.blackScholes(
      market.price, 
      this.STRIKE, 
      T, 
      this.targetVol, 
      this.RISK_FREE, 
      true
    );
    
    // Calculate fair price using realized volatility
    const fairPrice = pricingModels.blackScholes(
      market.price, 
      this.STRIKE, 
      T, 
      realized, 
      this.RISK_FREE, 
      true
    );
    
    // Check if option is underpriced relative to fair value (buy opportunity)
    if (optPrice < fairPrice * 0.97) {
      const qty = 10;
      if (this.cash > qty * optPrice) {
        this.options += qty;
        this.cash -= qty * optPrice;
        this.premium += qty * optPrice;
        this.logMsg(`VOL BUY 10 @ ${optPrice.toFixed(0)} (realVol ${(realized * 100).toFixed(1)}%)`);
      }
    } 
    // Check if option is overpriced relative to fair value (sell opportunity)
    else if (optPrice > fairPrice * 1.03) {
      const qty = 10;
      this.options -= qty;
      this.cash += qty * optPrice;
      this.premium -= qty * optPrice;
      this.logMsg(`VOL SELL 10 @ ${optPrice.toFixed(0)} (realVol ${(realized * 100).toFixed(1)}%)`);
    }
  }
}

module.exports = VolArb;