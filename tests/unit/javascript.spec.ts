import { test, expect } from '@playwright/test';

test.describe('JavaScript Functionality Tests', () => {
  test('should correctly implement Black-Scholes formula', async ({ page }) => {
    // Navigate to the page to access the JavaScript functions
    await page.goto('/fusion-trader.html');
    
    // Test the Black-Scholes implementation
    const optionPrice = await page.evaluate(() => {
      return blackScholes(50000, 50000, 7/365.25, 0.45, 0.03, true);
    });
    
    // The option price should be positive and reasonable for these parameters
    expect(optionPrice).toBeGreaterThan(0);
    expect(optionPrice).toBeLessThan(50000); // Should not be absurdly high
    
    // Test with different parameters
    const optionPrice2 = await page.evaluate(() => {
      return blackScholes(55000, 50000, 7/365.25, 0.45, 0.03, true);
    });
    
    // Higher stock price should give higher call option value
    expect(optionPrice2).toBeGreaterThan(optionPrice);
  });

  test('should correctly calculate delta', async ({ page }) => {
    // Navigate to the page to access the JavaScript functions
    await page.goto('/fusion-trader.html');
    
    // Test the delta calculation
    const callDelta = await page.evaluate(() => {
      return delta(50000, 50000, 7/365.25, 0.45, 0.03, true);
    });
    
    // Delta of at-the-money call should be around 0.5
    expect(callDelta).toBeGreaterThan(0.4);
    expect(callDelta).toBeLessThan(0.6);
    
    // Test put delta (should be negative)
    const putDelta = await page.evaluate(() => {
      return delta(50000, 50000, 7/365.25, 0.45, 0.03, false);
    });
    
    expect(putDelta).toBeLessThan(0);
    expect(putDelta).toBeGreaterThan(-0.6);
  });

  test('should generate random numbers with randn function', async ({ page }) => {
    // Navigate to the page to access the JavaScript functions
    await page.goto('/fusion-trader.html');
    
    // Generate multiple random numbers and check they're within reasonable bounds
    const randValues = await page.evaluate(() => {
      const values = [];
      for (let i = 0; i < 100; i++) {
        values.push(randn());
      }
      return values;
    });
    
    // Check that we got 100 values
    expect(randValues.length).toBe(100);
    
    // Check that all values are finite numbers (not NaN or Infinity)
    for (const val of randValues) {
      expect(Number.isFinite(val)).toBeTruthy();
    }
    
    // Check that the distribution seems reasonable (mean near 0, std dev near 1)
    const mean = randValues.reduce((a, b) => a + b, 0) / randValues.length;
    const variance = randValues.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / randValues.length;
    const stdDev = Math.sqrt(variance);
    
    // These are probabilistic checks - with 100 samples, they should generally pass
    // but might occasionally fail with very low probability
    expect(mean).toBeGreaterThan(-0.5);
    expect(mean).toBeLessThan(0.5);
    expect(stdDev).toBeGreaterThan(0.5);
    expect(stdDev).toBeLessThan(1.5);
  });

  test('should implement erf function correctly', async ({ page }) => {
    // Navigate to the page to access the JavaScript functions
    await page.goto('/fusion-trader.html');
    
    // Test some known values of the error function
    const erf0 = await page.evaluate(() => {
      return erf(0);
    });
    
    // erf(0) should be 0
    expect(Math.abs(erf0)).toBeLessThan(0.01);
    
    const erf1 = await page.evaluate(() => {
      return erf(1);
    });
    
    // erf(1) should be approximately 0.8427
    expect(erf1).toBeGreaterThan(0.8);
    expect(erf1).toBeLessThan(0.9);
    
    const erfNeg1 = await page.evaluate(() => {
      return erf(-1);
    });
    
    // erf should be odd function: erf(-x) = -erf(x)
    expect(Math.abs(erf1 + erfNeg1)).toBeLessThan(0.1);
  });
});