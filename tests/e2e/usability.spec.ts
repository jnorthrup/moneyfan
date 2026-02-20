import { test, expect } from '@playwright/test';

test.describe('Fusion Trader Usability Tests', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/fusion-trader.html');
  });

  test('should load the main page and display header', async ({ page }) => {
    // Check that the page loaded correctly
    await expect(page).toHaveTitle(/Fusion Trader/);
    
    // Verify the header is present
    const header = page.locator('header');
    await expect(header).toBeVisible();
    await expect(header).toContainText('Fusion Trader - Advanced Crypto Options Arena');
  });

  test('should display market information', async ({ page }) => {
    // Check that market info section is visible
    const marketInfo = page.locator('#market-info');
    await expect(marketInfo).toBeVisible();
    
    // Verify all market data elements are present
    await expect(page.locator('#current-price')).toBeVisible();
    await expect(page.locator('#current-vol')).toBeVisible();
    await expect(page.locator('#time-exp')).toBeVisible();
    await expect(page.locator('#total-pnl')).toBeVisible();
  });

  test('should display trading bots arena', async ({ page }) => {
    // Verify the arena section exists and contains bot elements
    const arena = page.locator('#arena');
    await expect(arena).toBeVisible();
    
    // Check that bot containers are present (there should be 5 initially)
    const botElements = page.locator('.bot');
    await expect(botElements).toHaveCount(5);
    
    // Verify each bot has the necessary elements
    await expect(page.locator('.bot').first().locator('h3')).toBeVisible();
    await expect(page.locator('.bot').first().locator('.log')).toBeVisible();
  });

  test('should have functional control buttons', async ({ page }) => {
    // Check that control buttons exist
    await expect(page.locator('#startBtn')).toBeVisible();
    await expect(page.locator('#resetBtn')).toBeVisible();
    await expect(page.locator('#addBotBtn')).toBeVisible();
    
    // Check speed slider
    await expect(page.locator('#speed')).toBeVisible();
    
    // Test adding a bot
    const initialBotCount = await page.locator('.bot').count();
    await page.locator('#addBotBtn').click();
    await expect(page.locator('.bot')).toHaveCount(initialBotCount + 1);
  });

  test('should start/pause simulation', async ({ page }) => {
    // Initially, the button should say "Start Battle"
    const startButton = page.locator('#startBtn');
    await expect(startButton).toContainText('Start Battle');
    
    // Click to start
    await startButton.click();
    await expect(startButton).toContainText('Pause');
    
    // Click to pause again
    await startButton.click();
    await expect(startButton).toContainText('Start Battle');
  });

  test('should reset simulation correctly', async ({ page }) => {
    // Start the simulation briefly
    await page.locator('#startBtn').click();
    await page.waitForTimeout(1000); // Run for a short time
    const startButton = page.locator('#startBtn');
    
    // Get initial state
    const initialTotalPnL = await page.locator('#total-pnl').textContent();
    
    // Pause and reset
    await startButton.click(); // Pause
    await page.locator('#resetBtn').click();
    
    // After reset, the PnL should be back to 0
    await expect(page.locator('#total-pnl')).toContainText('0.00');
    
    // Check that all bots are reset
    const botLogs = page.locator('.log');
    for (const botLog of await botLogs.all()) {
      await expect(botLog).toContainText('Cash: 1000000');
    }
  });

  test('should adjust simulation speed', async ({ page }) => {
    // Check initial speed value
    const speedSlider = page.locator('#speed');
    await expect(speedSlider).toHaveValue('500');
    
    // Change the speed
    await speedSlider.fill('1000');
    await expect(speedSlider).toHaveValue('1000');
  });

  test('should display bot logs correctly', async ({ page }) => {
    // Start the simulation briefly to generate logs
    await page.locator('#startBtn').click();
    await page.waitForTimeout(1500); // Wait for some bot actions
    
    // Stop the simulation
    await page.locator('#startBtn').click();
    
    // Check that bots have logs (each bot should have at least some content)
    const botLogs = page.locator('.bot .log');
    for (const botLog of await botLogs.all()) {
      await expect(botLog).not.toBeEmpty();
    }
  });
});