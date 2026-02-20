// src/api/routes.js - API routes with access to market and bot managers
// This file is now a function that accepts market and botManager instances

function createRoutes(market, botManager) {
  const express = require('express');
  const router = express.Router();
  const KotlinBridge = require('../utils/kotlinBridge');
  const kotlinBridge = new KotlinBridge();

  router.get('/status', (req, res) => {
    res.json({ 
      status: 'Server is running', 
      timestamp: new Date().toISOString(),
      isRunning: market.isRunning,
      marketPrice: market.price
    });
  });

  router.get('/market', (req, res) => {
    res.json(market.getState());
  });

  router.get('/bots', (req, res) => {
    res.json(botManager.getStates(market.price));
  });

  router.post('/control/start', (req, res) => {
    const speed = req.body.speed || 500;
    market.start(speed);
    res.json({ message: 'Simulation started', speed });
  });

  router.post('/control/stop', (req, res) => {
    market.stop();
    res.json({ message: 'Simulation stopped' });
  });

  router.post('/control/reset', (req, res) => {
    market.reset();
    botManager.reset();
    res.json({ message: 'Simulation reset' });
  });

  // Kotlin integration endpoints
  router.get('/kotlin/status', async (req, res) => {
    try {
      const isAvailable = await kotlinBridge.checkKotlinAvailability();
      res.json({ 
        kotlinAvailable: isAvailable,
        scriptsPath: kotlinBridge.kotlinScriptsPath,
        mainBotScript: kotlinBridge.mainBotScript,
        minimalBotScript: kotlinBridge.minimalBotScript
      });
    } catch (error) {
      res.status(500).json({ error: error.message });
    }
  });

  router.get('/kotlin/test-minimal', async (req, res) => {
    try {
      const result = await kotlinBridge.testMinimalBot();
      res.json(result);
    } catch (error) {
      res.status(500).json({ error: error.message });
    }
  });

  router.get('/kotlin/main-bot-status', async (req, res) => {
    try {
      const result = await kotlinBridge.getMainBotStatus();
      res.json(result);
    } catch (error) {
      res.status(500).json({ error: error.message });
    }
  });

  router.post('/kotlin/start-main-bot', async (req, res) => {
    try {
      const result = await kotlinBridge.startMainBot();
      res.json(result);
    } catch (error) {
      res.status(500).json({ error: error.message });
    }
  });

  return router;
}

module.exports = createRoutes;