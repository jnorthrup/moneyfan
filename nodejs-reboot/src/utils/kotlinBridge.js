// src/utils/kotlinBridge.js - Bridge to interface with existing Kotlin scripts
const { spawn } = require('child_process');
const fs = require('fs').promises;
const path = require('path');

class KotlinBridge {
  constructor() {
    // Define paths to the Kotlin scripts
    this.kotlinScriptsPath = path.join(__dirname, '../../../');
    this.mainBotScript = path.join(this.kotlinScriptsPath, 'CoinbaseXChangeBotMain.kts');
    this.minimalBotScript = path.join(this.kotlinScriptsPath, 'CoinbaseXChangeBotMinimal.kts');
  }

  // Check if Kotlin is installed and available
  async checkKotlinAvailability() {
    return new Promise((resolve) => {
      const child = spawn('which', ['kscript']);
      child.on('close', (code) => {
        resolve(code === 0);
      });
      
      // Fallback: check for Kotlin
      if (child.exitCode === 1) {
        const child2 = spawn('which', ['kotlin']);
        child2.on('close', (code) => {
          resolve(code === 0);
        });
      }
    });
  }

  // Run a Kotlin script and capture output
  async runKotlinScript(scriptPath, args = []) {
    // Check if the script exists
    try {
      await fs.access(scriptPath);
    } catch (error) {
      throw new Error(`Kotlin script not found: ${scriptPath}`);
    }

    // Execute the script using kscript
    return new Promise((resolve, reject) => {
      const command = 'kscript';
      const params = [scriptPath, ...args];
      
      const childProcess = spawn(command, params);
      let output = '';
      let errorOutput = '';

      childProcess.stdout.on('data', (data) => {
        output += data.toString();
      });

      childProcess.stderr.on('data', (data) => {
        errorOutput += data.toString();
      });

      childProcess.on('close', (code) => {
        if (code === 0) {
          resolve({
            success: true,
            output: output,
            error: errorOutput
          });
        } else {
          reject(new Error(`Script execution failed with code ${code}: ${errorOutput}`));
        }
      });
    });
  }

  // Get status of the main trading bot
  async getMainBotStatus() {
    try {
      const result = await this.runKotlinScript(this.mainBotScript, ['--status']);
      return {
        status: 'running',
        details: result.output
      };
    } catch (error) {
      // If the status check fails, the bot may not be running
      return {
        status: 'stopped',
        error: error.message
      };
    }
  }

  // Start the main trading bot
  async startMainBot() {
    try {
      // In a real implementation, we would start the bot and monitor its process
      // For now, we'll just validate that the script exists and can be executed
      await fs.access(this.mainBotScript);
      return {
        success: true,
        message: 'Main bot is ready to start. See server logs for execution details.'
      };
    } catch (error) {
      return {
        success: false,
        error: `Main bot unavailable: ${error.message}`
      };
    }
  }

  // Execute a test run of the minimal bot
  async testMinimalBot() {
    try {
      const result = await this.runKotlinScript(this.minimalBotScript);
      return {
        success: true,
        output: result.output
      };
    } catch (error) {
      return {
        success: false,
        error: error.message
      };
    }
  }
}

module.exports = KotlinBridge;