const express = require('express');
const http = require('http');
const socketIo = require('socket.io');
const cors = require('cors');
const path = require('path');

require('dotenv').config();

const app = express();
const server = http.createServer(app);
const io = socketIo(server, {
  cors: {
    origin: "*",
    methods: ["GET", "POST"]
  }
});

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

// Import market engine and bot manager
const MarketEngine = require('./src/market/marketEngine');
const BotManager = require('./src/bots/botManager');

// Initialize market and bots
const market = new MarketEngine();
const botManager = new BotManager();

// Import API routes (with market and bot manager instances)
const createApiRoutes = require('./src/api/routes');
const apiRoutes = createApiRoutes(market, botManager);
app.use('/api', apiRoutes);

// Serve the main HTML file
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Socket.IO connection handling
io.on('connection', (socket) => {
  console.log('New client connected:', socket.id);

  // Send initial market and bot data to new clients
  socket.emit('marketUpdate', market.getState());
  socket.emit('botsUpdate', botManager.getStates(market.price));

  socket.on('control:start', (data) => {
    const speed = data.speed || 500;
    market.start(speed);
    io.emit('statusUpdate', { message: 'Simulation started', isRunning: true });
  });

  socket.on('control:stop', () => {
    market.stop();
    io.emit('statusUpdate', { message: 'Simulation stopped', isRunning: false });
  });

  socket.on('control:reset', () => {
    market.reset();
    botManager.reset();
    io.emit('marketUpdate', market.getState());
    io.emit('botsUpdate', botManager.getStates(market.price));
    io.emit('statusUpdate', { message: 'Simulation reset', isRunning: false });
  });

  socket.on('requestInitialData', () => {
    socket.emit('marketUpdate', market.getState());
    socket.emit('botsUpdate', botManager.getStates(market.price));
    socket.emit('statusUpdate', { 
      message: 'Initial data sent', 
      isRunning: market.isRunning 
    });
  });

  socket.on('disconnect', () => {
    console.log('Client disconnected:', socket.id);
  });
});

// Set up the simulation loop to broadcast updates
setInterval(() => {
  if (market.isRunning) {
    market.step();
    botManager.updateAll(market);
    
    // Broadcast updated market and bot data to all clients
    io.emit('marketUpdate', market.getState());
    io.emit('botsUpdate', botManager.getStates(market.price));
  }
}, market.speed);

const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
  console.log(`MoneyFan Node.js server running on port ${PORT}`);
});