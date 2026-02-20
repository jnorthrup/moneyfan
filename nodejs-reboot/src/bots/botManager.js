// botManager.js - Manages all trading bots
const DeltaHedger = require('./deltaHedger');
const VolArb = require('./volArb');
const Momentum = require('./momentum');

class BotManager {
  constructor() {
    this.bots = [
      new DeltaHedger("DeltaHedgeBot", "#0af"),
      new VolArb("VolArbBot", "#f90"),
      new Momentum("MomBot", "#3c3")
    ];
  }

  // Update all bots with current market state
  updateAll(market) {
    this.bots.forEach(bot => {
      bot.update(market);
    });
  }

  // Get states of all bots
  getStates(marketPrice) {
    return this.bots.map(bot => bot.getState(marketPrice));
  }

  // Reset all bots to initial state
  reset() {
    this.bots = [
      new DeltaHedger("DeltaHedgeBot", "#0af"),
      new VolArb("VolArbBot", "#f90"),
      new Momentum("MomBot", "#3c3")
    ];
  }

  // Get bot by name
  getBotByName(name) {
    return this.bots.find(bot => bot.name === name);
  }
}

module.exports = BotManager;