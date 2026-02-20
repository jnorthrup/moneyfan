"""
Run Historical Training Session - HRM Metrics + Winning Agent Stats
===================================================================

This script runs the historical training session showing:
1. HRM learning/loss metrics
2. Winning agent statistics (DD, P&L, Loss, etc.)
3. 24-hour visualization of training progress
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from historical_training_session import HistoricalTrainingSession, HistoricalTrainingConfig
from datetime import datetime

def main():
    print("HISTORICAL TRAINING SESSION - HRM METRICS + WINNING AGENT STATS")
    print("="*80)
    print()
    
    # Create configuration for comprehensive training session
    config = HistoricalTrainingConfig(
        symbol="BTC-USD",
        timeframe="1h",
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 8),
        epochs=100,  # Full training session
        session_duration_hours=24,
        update_frequency_seconds=1,  # Fast updates for visibility
        data_source="synthetic",  # Use synthetic data for demo
    )
    
    print("CONFIGURATION:")
    print(f"  Symbol: {config.symbol}")
    print(f"  Timeframe: {config.timeframe}")
    print(f"  Training Period: {config.start_date.date()} to {config.end_date.date()}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Update Frequency: Every {config.update_frequency_seconds} second(s)")
    print()
    
    print("METRICS DISPLAYED:")
    print("  HRM Learning Metrics:")
    print("    - Loss (HRM learning loss)")
    print("    - Accuracy (Prediction accuracy)")
    print("    - HRM Reward (Internal reward signal)")
    print("    - Veto Rate (Trade veto rate)")
    print("    - Regime Confidence (Market regime confidence)")
    print()
    print("  Winning Agent Statistics:")
    print("    - Wins/Losses (Trade counts)")
    print("    - Avg Win/Avg Loss (Average P&L per trade)")
    print("    - Largest Win/Loss (Extreme outcomes)")
    print("    - Expected Value ($ per trade expectancy)")
    print("    - Recovery Factor (P&L / Max DD)")
    print("    - Risk Adjusted Return (Sharpe-like metric)")
    print()
    print("  Drawdown & Streak Metrics:")
    print("    - Drawdown Depth (% peak-to-trough)")
    print("    - Drawdown Duration (time to recovery)")
    print("    - Consecutive Wins/Losses (current streak)")
    print("    - Win/Loss Streak (max streak)")
    print()
    print("  Trader Performance Metrics:")
    print("    - Total Trades")
    print("    - Win Rate (%)")
    print("    - Profit Factor (Gross Profit / Gross Loss)")
    print("    - Sharpe Ratio")
    print("    - Max Drawdown (%)")
    print("    - Total P&L ($)")
    print("    - Final Equity ($)")
    print()
    
    # Create and run session
    session = HistoricalTrainingSession(config)
    
    print("STARTING TRAINING SESSION...")
    print("="*80)
    print()
    
    session_dir = session.run()
    
    if session_dir:
        print()
        print("="*80)
        print("TRAINING SESSION COMPLETED SUCCESSFULLY!")
        print("="*80)
        print()
        print(f"Results saved to: {session_dir}")
        print()
        print("FILES CREATED:")
        print("  - training_metrics.json (Complete metrics history)")
        print("  - training_progress.png (All metrics visualization)")
        print("  - equity_curve.png (Detailed equity curve)")
        print()
        print("NEXT STEPS:")
        print("  1. Review the training metrics in training_metrics.json")
        print("  2. Examine the plots in the session directory")
        print("  3. Use these metrics for HRM optimization")
        print("  4. Compare against baseline strategies")
        print()
    else:
        print("❌ Training session failed")

if __name__ == "__main__":
    main()