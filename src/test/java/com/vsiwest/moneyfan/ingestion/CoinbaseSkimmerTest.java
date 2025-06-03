package com.vsiwest.moneyfan.ingestion;

import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.junit.jupiter.api.Assertions.assertNotNull;

class CoinbaseSkimmerTest {

    private static final Logger logger = LoggerFactory.getLogger(CoinbaseSkimmerTest.class);

    @Test
    void skimmerClassCanBeLoaded() {
        // This is a very basic test to ensure the class can be loaded.
        // It does not test the main method's logic due to its static nature,
        // infinite loop, and direct System.getenv() calls which are hard to unit test
        // without significant refactoring or more complex testing tools.
        CoinbaseSkimmer skimmer = new CoinbaseSkimmer(); // Should be instantiable if it has a default constructor
        assertNotNull(skimmer, "CoinbaseSkimmer instance should not be null");
        logger.info("CoinbaseSkimmer class loaded. Note: main() logic is not directly tested here.");
    }

    @Test
    @Disabled("Testing main() directly is complex and better handled by integration/system tests or by refactoring main logic into testable components.")
    void mainMethodExecutionTest() {
        // This test is disabled because:
        // 1. main() is static and calls System.exit() or runs an infinite loop.
        // 2. It directly instantiates CoinbaseApiConfig which uses System.getenv().
        // 3. CoinbaseApiConfig and CoinbaseApiClient are tested in their own unit tests.
        // A proper test would require refactoring CoinbaseSkimmer or using tools like System Rules / PowerMock.
        // For now, we rely on component-level tests and potential future integration tests.
        logger.warn("Test for main() method is disabled. See comments in test for details.");
    }

    // Further tests would require refactoring CoinbaseSkimmer.main()
    // to make its core logic testable by injecting dependencies or
    // separating the loop from the setup.
    // For example:
    // - Test what happens if CoinbaseApiConfig throws IllegalStateException (covered in CoinbaseApiConfigTest)
    // - Test what happens if CoinbaseApiClient.getAccountBalances throws CoinbaseApiException (covered in CoinbaseApiClientTest)
}
