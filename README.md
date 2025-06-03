# Coinbase Skimmer

This application is a simple skimmer for Coinbase, designed to connect to the Coinbase Advanced Trade API, fetch account balances, and potentially be extended for further market interaction.

## Prerequisites

*   Java JDK 21 or later
*   Apache Maven 3.6.x or later
*   Active Coinbase API Key and Secret with appropriate permissions (e.g., view balances).

## Configuration

The application requires Coinbase API credentials to be set as environment variables:

*   `COINBASE_API_KEY`: Your Coinbase API Key.
*   `COINBASE_API_SECRET`: Your Coinbase API Secret.

Example:
```bash
export COINBASE_API_KEY="your_api_key_here"
export COINBASE_API_SECRET="your_api_secret_here"
```

### Logging Configuration

Logging is configured via SLF4J with `slf4j-simple`. The configuration file is located at `src/main/resources/simplelogger.properties`. By default, it logs INFO messages to the console, with DEBUG level for application-specific packages. You can customize this file to change log levels, formats, or output destinations.

## Building the Application

To build the application, navigate to the project root directory and run:

```bash
./mvnw clean package
```
This will compile the code, run tests, and create an executable JAR file in the `target/` directory (e.g., `moneyfan-1.0.0-SNAPSHOT.jar`).

## Running the Application

Once built, you can run the application using:

```bash
java -jar target/moneyfan-1.0.0-SNAPSHOT.jar
```
Ensure the environment variables for API keys are set in the shell session where you run this command. The application will start fetching and displaying your Coinbase account balances periodically.

To run directly using Maven (after setting environment variables):
```bash
./mvnw exec:java
```

## Development

*   **Main Class**: `com.vsiwest.moneyfan.ingestion.CoinbaseSkimmer`
*   **API Client**: `com.vsiwest.moneyfan.coinbase.CoinbaseApiClient`
*   **Configuration**: `com.vsiwest.moneyfan.config.CoinbaseApiConfig`

### Running Tests
```bash
./mvnw test
```
