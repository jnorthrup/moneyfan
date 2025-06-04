# Coinbase Trading Bot (Kotlin Script)

This repository contains a Kotlin script (`coinbaseXChangeBot.main.kts`) that implements an automated trading bot for the Coinbase exchange (specifically targeting the Advanced Trade API). It is a port and adaptation of a strategy originally designed for the Kraken exchange.

The bot utilizes the Knowm XChange library for interacting with Coinbase APIs and incorporates several configurable trading strategies, including asset harvesting, portfolio rebalancing, Adaptive Dead Zone (ADZ) adjustments, and Crash Protection (CP) mechanisms. The script is designed to be run using `kscript`.

## Core Features

*   **Coinbase Integration**: Interacts with Coinbase Advanced Trade API via the XChange library.
*   **Standalone Script**: Entire bot logic is contained in a single `.kts` file.
*   **Real-time Price Feeds**: Subscribes to WebSocket price tickers for specified trading pairs.
*   **State Persistence**: Saves and loads its operational state (asset baselines, strategy states, timestamps) to a `coinbaseBotState.json` file, allowing it to resume across sessions.
*   **Configurable Strategies**:
    *   Individual Asset Harvesting (profit taking).
    *   Portfolio Override Harvesting (portfolio-level baseline reset).
    *   Harvest Proceeds Allocation (reinvest, BTC buy placeholder, cash).
    *   Asset Rebalancing (buying back into underperforming assets).
    *   Adaptive Dead Zone (ADZ) to dynamically adjust strategy triggers.
    *   Crash Protection (CP) to modify strategy parameters during market downturns.
*   **Simulated Trading by Default**: **Crucially, all trading actions are SIMULATED by default and logged to the console. No real orders are placed without modifying the script.**
*   **Ad-hoc Tensor-Core Inspired Types**: Uses internal, ad-hoc definitions for `Series` and `Tensor`-like data structures for some data representations and calculations, inspired by TrikeShed/Tensor-Core principles.

## Prerequisites

1.  **Java Development Kit (JDK)**: Version 11 or later is recommended. Kotlin and XChange run on the JVM.
2.  **Kotlin Compiler**: While `kscript` often bundles or manages this, having a standalone Kotlin compiler can be useful.
3.  **`kscript`**: This is the recommended way to run the script. `kscript` makes running Kotlin scripts much easier by handling dependency resolution and compilation on the fly.
    *   **Installation via SDKMAN! (Recommended)**:
        ```bash
        curl -s "https://get.sdkman.io" | bash
        source "$HOME/.sdkman/bin/sdkman-init.sh"
        sdk install kotlin
        sdk install kscript
        ```
    *   **Installation via Homebrew (macOS)**:
        ```bash
        brew install kotlin
        brew install kscript
        ```
    *   **Manual Installation**: Download the `kscript` binary from the [kscript GitHub releases page](https://github.com/kscripting/kscript/releases) and place it in your PATH.

## Dependency Management

This script is self-contained. All external library dependencies (XChange, Kotlinx Serialization, SLF4J, RxJava, Jackson, etc.) are declared at the top of the `coinbaseXChangeBot.main.kts` file using `@file:DependsOn` annotations. `kscript` will automatically download and cache these dependencies the first time the script is run.

## Configuration

1.  **API Credentials (Environment Variables)**:
    You **MUST** set the following environment variables in your shell session before running the script:
    *   `COINBASE_API_KEY`: Your Coinbase API Key.
    *   `COINBASE_API_SECRET`: Your Coinbase API Secret.

    These keys need permissions for viewing balances, market data, and (if you enable live trading) placing trades on the Coinbase Advanced Trade platform.

    Example:
    ```bash
    export COINBASE_API_KEY="your_coinbase_api_key"
    export COINBASE_API_SECRET="your_coinbase_api_secret"
    ```

2.  **Internal Script Parameters**:
    Many strategy parameters (trigger percentages, timeouts, allocation ratios, feature enable/disable flags like `ENABLE_ADAPTIVE_DEAD_ZONE`, etc.) are defined as `val` or `const val` at the top of the `coinbaseXChangeBot.main.kts` script (e.g., `FLAT_HARVEST_TRIGGER_PERCENT`, `REBALANCE_COOLDOWN`). You can modify these values directly in the script to tune the bot's behavior.

## Running the Bot

1.  **Ensure Prerequisites**: Verify JDK, Kotlin, and `kscript` are installed.
2.  **Set Environment Variables**: Export your `COINBASE_API_KEY` and `COINBASE_API_SECRET`.
3.  **Make Script Executable (Optional but Recommended)**:
    If you add `#!/usr/bin/env kscript` as the very first line of the script:
    ```bash
    chmod +x coinbaseXChangeBot.main.kts
    ```
4.  **Execute the Script**:
    *   Using `kscript` directly:
        ```bash
        kscript coinbaseXChangeBot.main.kts
        ```
    *   If made executable with the shebang:
        ```bash
        ./coinbaseXChangeBot.main.kts
        ```

The script will start, initialize the connection to Coinbase, begin fetching market data, and enter its main operational loop, logging its actions to the console.

## Operational Notes

### **VERY IMPORTANT: Simulated Trading Mode**

The script, as provided in this repository, runs in a **SIMULATED TRADING MODE**.
*   All BUY and SELL order placements are currently **placeholders**.
*   Instead of executing real trades, the script logs messages like:
    `SIMULATED Portfolio Harvest SELL for BTC...`
    `SIMULATED Reinvest BUY for ETH...`
*   **NO REAL TRADES WILL BE PLACED, AND NO REAL FUNDS ARE AT RISK WITH THE SCRIPT IN ITS CURRENT STATE.**

To enable live trading, you would need to:
1.  Carefully review the code sections responsible for placing orders (primarily within `ExchangeService.placeMarketOrder` and how it's called by the strategy logic).
2.  Replace the simulation placeholders with actual calls to `tradeService.placeMarketOrder(marketOrder)` from the XChange library.
3.  **This should only be done after extensive testing, understanding all risks involved, and potentially starting with very small amounts of capital.**

### State File

The bot persists its operational state (asset baselines, strategy-specific states like trailing and rebalancing info, last action timestamps) in a JSON file named `coinbaseBotState.json`. This file is created and updated in the same directory where the script is run. Deleting this file will cause the bot to start with a fresh state, re-initializing baselines.

### Logging

*   Logs are output to the console using SLF4J with a simple binding (`slf4j-simple`).
*   Log levels for different components are set at the beginning of the `main` function in the script using `System.setProperty("org.slf4j.simpleLogger.log.component", "level")`. You can adjust these for more or less verbosity.
*   For more advanced logging configuration (e.g., writing to a file), you can create a `simplelogger.properties` file in the same directory as the script. Refer to the [SLF4J SimpleLogger documentation](https://www.slf4j.org/manual.html#simple) for configuration options.

## Disclaimer

**TRADING CRYPTOCURRENCIES INVOLVES SUBSTANTIAL RISK OF LOSS AND IS NOT SUITABLE FOR EVERY INVESTOR. ALL TRADING DECISIONS ARE YOUR OWN RESPONSIBILITY.**

This script is provided for **educational and experimental purposes only**. The strategies implemented are based on a pre-existing bot and have not been professionally audited or guaranteed for profit. Market conditions can change rapidly, and past performance is not indicative of future results.

**USE THIS SCRIPT AT YOUR OWN RISK. The authors and contributors are not responsible for any financial losses or other damages incurred from its use.** Always start with simulated trading and, if you choose to trade live, use only funds you can afford to lose.
```
