# Tech Stack: moneyfan-trader

## Languages
- **JavaScript / Node.js (ESM)** — primary trading engine, backtesting, web server
- **Python 3.14** — ML signal pipeline (`hrm/`), auth research scripts
- **Kotlin 2.1 (JVM)** — alternative exchange bot via XChange library

## Frontend
- **Vite 5** — build tool and dev server for the trading dashboard
- **HTML/CSS** — dashboard UI (`src/main/resources/fusion-trader.html`)

## Backend / Runtime
- **Node.js** — primary runtime (ESM modules)
- **Express 5** — HTTP server for web dashboard and API
- **WebSockets (`ws`)** — real-time price feed consumption

## Databases
- **SQLite** (`better-sqlite3`) — local trade history and state persistence
- **DuckDB** — analytical backtesting queries over OHLCV datasets

## Exchange Integration
- **Coinbase Advanced Trade API** — primary exchange (REST + WebSocket, public + authenticated)
- **Kraken** — secondary exchange via XChange (knowm `xchange-core` 5.2.2)
- **XChange / knowm** — JVM-based exchange abstraction layer (Maven)

## ML / Signal Pipeline
- **PyTorch** — HRM model training and inference (`hrm/` module)
- **Python** — continuous training, A/B testing, checkpoint management

## Testing
- **Playwright** — E2E usability tests and JavaScript unit tests (cross-browser: Chromium, Firefox, WebKit)
- **Node.js test runner** — unit tests for trading logic

## Build & Package Management
- **pnpm** — Node.js package manager
- **Maven 3** — JVM build tool for Kotlin bot

## Utilities
- **dotenv** — environment variable management (API keys, secrets)
- **chalk** — terminal output formatting
- **sparkly** — terminal sparkline charts
