#!/usr/bin/env python3
"""
Quick Variability Check - Simple Feedback Loop
===============================================

Quick analysis of pretraining loss variability for model sizing feedback.

Usage:
    python quick_variability_check.py
"""

import argparse
import re
import numpy as np


def extract_losses(log_file: str = "train_pretrain_stochastic_continue.log") -> list:
    """Extract losses from log file"""
    try:
        with open(log_file, 'r') as f:
            content = f.read()
        
        pattern = r'pred_loss=([\d.]+)'
        matches = re.findall(pattern, content)
        return [float(m) for m in matches]
    except Exception as e:
        print(f"Error: {e}")
        return []


def analyze_variability(losses: list) -> dict:
    """Quick variability analysis"""
    if not losses:
        return {}
    
    arr = np.array(losses)
    return {
        'n': len(losses),
        'mean': np.mean(arr),
        'std': np.std(arr),
        'cv': np.std(arr) / np.mean(arr),
        'min': np.min(arr),
        'max': np.max(arr),
        'median': np.median(arr),
        'q1': np.percentile(arr, 25),
        'q3': np.percentile(arr, 75),
    }


def get_recommendation(analysis: dict) -> str:
    """Get model sizing recommendation"""
    if not analysis:
        return "No data"
    
    mean = analysis['mean']
    cv = analysis['cv']
    max_loss = analysis['max']
    
    if mean > 400 and max_loss > 800:
        return "🔥 SEVERELY UNDERPOWERED - Increase hidden_dim to 128-256"
    elif mean > 200 and max_loss > 500:
        return "⚠️ UNDERPOWERED - Consider testing hidden_dim=128"
    elif mean < 50 and max_loss < 100:
        return "✅ OPTIMAL/SLIGHTLY OVERPOWERED"
    elif cv > 0.5:
        return "📈 HIGH VARIABILITY - Add more layers (3-4 each)"
    elif cv < 0.2:
        return "📉 LOW VARIABILITY - Model converging well"
    else:
        return "✅ MODERATE - Current config may be adequate"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Analyze pretraining loss variability from a real log file.")
    parser.add_argument(
        "--log-file",
        default="train_pretrain_stochastic_continue.log",
        help="Training log file containing pred_loss=... lines.",
    )
    args = parser.parse_args(argv)

    losses = extract_losses(args.log_file)
    if not losses:
        print(f"No loss data found in {args.log_file}; refusing to fabricate sizing guidance.")
        return 1

    analysis = analyze_variability(losses)
    
    # Print quick analysis
    print("\n" + "="*60)
    print("QUICK VARIABILITY CHECK - MODEL SIZING FEEDBACK")
    print("="*60)
    
    if analysis:
        print(f"\n📊 DATA: {analysis['n']} episodes analyzed")
        print(f"   Mean loss: {analysis['mean']:.1f}")
        print(f"   Variability (CV): {analysis['cv']:.3f}")
        print(f"   Range: {analysis['min']:.1f} - {analysis['max']:.1f}")
        print(f"   Median: {analysis['median']:.1f}")
        
        print(f"\n🎯 RECOMMENDATION:")
        print(f"   {get_recommendation(analysis)}")
        
        print(f"\n📋 QUICK GUIDE:")
        print(f"   CV < 0.2: Model too small/overfitting")
        print(f"   CV 0.2-0.4: Model well-sized")
        print(f"   CV 0.4-0.6: Model borderline")
        print(f"   CV > 0.6: Model too small/underfitting")
        
        print(f"\n💡 NEXT STEPS:")
        print(f"   1. If CV > 0.5: Test hidden_dim=128")
        print(f"   2. If mean loss > 400: Test hidden_dim=256")
        print(f"   3. If mean loss < 100: Model may be overpowered")
        
    print("\n" + "="*60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
