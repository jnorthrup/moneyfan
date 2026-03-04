#!/usr/bin/env python3
"""
Variability Feedback Loop for Stochastic Pretraining Analysis
==============================================================

Analyzes pretraining loss data to determine optimal variability rates
and provides recommendations for model sizing adjustments.

Usage:
    python variability_feedback_loop.py --log train_pretrain_stochastic_continue.log
    python variability_feedback_loop.py --data "301.6539,236.6157,245.4642,582.2495,..."
"""

import re
import numpy as np
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class VariabilityAnalysis:
    """Analysis results for pretraining loss variability"""
    losses: List[float]
    mean: float
    std: float
    cv: float  # Coefficient of variation
    range: Tuple[float, float]
    iqr: Tuple[float, float]
    outliers: List[float]
    trend: float  # Slope of loss over time
    stability_score: float  # 0-100, higher is more stable
    recommendation: str


class VariabilityFeedbackLoop:
    """Analyzes pretraining loss variability and provides model sizing recommendations"""
    
    def __init__(self):
        self.optimal_ranges = {
            'low_variability': {'cv': 0.1, 'max_loss': 50, 'stability_score': 80},
            'moderate_variability': {'cv': 0.3, 'max_loss': 200, 'stability_score': 60},
            'high_variability': {'cv': 0.5, 'max_loss': 500, 'stability_score': 40},
            'extreme_variability': {'cv': 0.7, 'max_loss': 1000, 'stability_score': 20}
        }
    
    def extract_losses_from_log(self, log_path: str) -> List[float]:
        """Extract pretraining losses from log file"""
        try:
            with open(log_path, 'r') as f:
                content = f.read()
            
            # Pattern to extract pred_loss values
            pattern = r'pred_loss=([\d.]+)'
            matches = re.findall(pattern, content)
            
            if not matches:
                print(f"No pred_loss found in {log_path}")
                return []
            
            losses = [float(match) for match in matches]
            print(f"Extracted {len(losses)} loss values from {log_path}")
            return losses
            
        except FileNotFoundError:
            print(f"Log file not found: {log_path}")
            return []
        except Exception as e:
            print(f"Error reading log file: {e}")
            return []
    
    def parse_losses_from_data(self, data_str: str) -> List[float]:
        """Parse losses from comma-separated string"""
        try:
            values = data_str.split(',')
            losses = [float(v.strip()) for v in values if v.strip()]
            print(f"Parsed {len(losses)} loss values from data string")
            return losses
        except Exception as e:
            print(f"Error parsing data: {e}")
            return []
    
    def calculate_variability_metrics(self, losses: List[float]) -> Dict:
        """Calculate comprehensive variability metrics"""
        if not losses:
            return {}
        
        losses_array = np.array(losses)
        
        # Basic statistics
        mean_loss = np.mean(losses_array)
        std_loss = np.std(losses_array)
        cv = std_loss / mean_loss if mean_loss > 0 else 0  # Coefficient of variation
        
        # Range and IQR
        min_loss = np.min(losses_array)
        max_loss = np.max(losses_array)
        q1 = np.percentile(losses_array, 25)
        q3 = np.percentile(losses_array, 75)
        iqr = q3 - q1
        
        # Outliers (using IQR method)
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = [x for x in losses_array if x < lower_bound or x > upper_bound]
        
        # Trend analysis (linear regression)
        if len(losses_array) > 1:
            x = np.arange(len(losses_array))
            slope, _ = np.polyfit(x, losses_array, 1)
            trend = float(slope)
        else:
            trend = 0.0
        
        # Stability score (0-100)
        # Lower CV and lower max loss = higher stability
        stability_score = max(0, 100 - (cv * 100) - (min(max_loss / 100, 50)))
        
        return {
            'mean': mean_loss,
            'std': std_loss,
            'cv': cv,
            'min': min_loss,
            'max': max_loss,
            'q1': q1,
            'q3': q3,
            'iqr': iqr,
            'outliers': outliers,
            'num_outliers': len(outliers),
            'outlier_pct': (len(outliers) / len(losses_array)) * 100,
            'trend': trend,
            'stability_score': stability_score
        }
    
    def analyze_variability(self, losses: List[float]) -> VariabilityAnalysis:
        """Comprehensive analysis of loss variability"""
        if not losses:
            return VariabilityAnalysis(
                losses=[], mean=0, std=0, cv=0, range=(0, 0),
                iqr=(0, 0), outliers=[], trend=0, stability_score=0,
                recommendation="No data to analyze"
            )
        
        metrics = self.calculate_variability_metrics(losses)
        
        # Determine variability category
        if metrics['cv'] < 0.2 and metrics['max'] < 100:
            category = "low_variability"
        elif metrics['cv'] < 0.4 and metrics['max'] < 300:
            category = "moderate_variability"
        elif metrics['cv'] < 0.6 and metrics['max'] < 600:
            category = "high_variability"
        else:
            category = "extreme_variability"
        
        # Generate recommendation
        recommendation = self.generate_recommendation(metrics, category)
        
        return VariabilityAnalysis(
            losses=losses,
            mean=metrics['mean'],
            std=metrics['std'],
            cv=metrics['cv'],
            range=(metrics['min'], metrics['max']),
            iqr=(metrics['q1'], metrics['q3']),
            outliers=metrics['outliers'],
            trend=metrics['trend'],
            stability_score=metrics['stability_score'],
            recommendation=recommendation
        )
    
    def generate_recommendation(self, metrics: Dict, category: str) -> str:
        """Generate model sizing recommendations based on variability"""
        
        recommendations = []
        
        # Check for extreme outliers
        if metrics['outlier_pct'] > 20:
            recommendations.append(
                f"⚠️ HIGH OUTLIER RATE: {metrics['outlier_pct']:.1f}% episodes have extreme losses"
            )
        
        # Check for negative trend (losses decreasing)
        if metrics['trend'] < -5:
            recommendations.append(
                f"✅ POSITIVE TREND: Losses decreasing at rate {metrics['trend']:.1f}/episode"
            )
        elif metrics['trend'] > 5:
            recommendations.append(
                f"❌ NEGATIVE TREND: Losses increasing at rate {metrics['trend']:.1f}/episode"
            )
        
        # Check stability score
        if metrics['stability_score'] < 30:
            recommendations.append(
                f"❌ UNSTABLE: Stability score {metrics['stability_score']:.1f}/100"
            )
        elif metrics['stability_score'] < 60:
            recommendations.append(
                f"⚠️ MODERATE: Stability score {metrics['stability_score']:.1f}/100"
            )
        else:
            recommendations.append(
                f"✅ STABLE: Stability score {metrics['stability_score']:.1f}/100"
            )
        
        # Model sizing recommendations
        if metrics['mean'] > 400 and metrics['max'] > 800:
            recommendations.append(
                "🔥 MODEL UNDERFITTING: Increase hidden_dim to 128-256"
            )
        elif metrics['mean'] > 200 and metrics['max'] > 500:
            recommendations.append(
                "⚠️ MODEL BORDERLINE: Consider testing hidden_dim=128"
            )
        elif metrics['mean'] < 50 and metrics['max'] < 100:
            recommendations.append(
                "✅ MODEL OVERFITTING: Consider reducing hidden_dim or adding dropout"
            )
        
        # Variability recommendations
        if metrics['cv'] > 0.5:
            recommendations.append(
                f"📈 HIGH VARIABILITY: CV={metrics['cv']:.2f} - Model struggles with diverse patterns"
            )
        elif metrics['cv'] < 0.2:
            recommendations.append(
                f"📉 LOW VARIABILITY: CV={metrics['cv']:.2f} - Model converging on specific patterns"
            )
        
        return "\n".join(recommendations) if recommendations else "✅ Model appears well-sized"
    
    def print_detailed_analysis(self, analysis: VariabilityAnalysis):
        """Print detailed variability analysis"""
        print("\n" + "="*80)
        print("VARIABILITY FEEDBACK LOOP ANALYSIS")
        print("="*80)
        
        print(f"\n📊 DATA SUMMARY:")
        print(f"  • Episodes analyzed: {len(analysis.losses)}")
        print(f"  • Mean loss: {analysis.mean:.2f}")
        print(f"  • Std deviation: {analysis.std:.2f}")
        print(f"  • Coefficient of variation: {analysis.cv:.3f}")
        
        print(f"\n📈 LOSS RANGE:")
        print(f"  • Minimum: {analysis.range[0]:.2f}")
        print(f"  • Maximum: {analysis.range[1]:.2f}")
        print(f"  • Range: {analysis.range[1] - analysis.range[0]:.2f}")
        
        print(f"\n📍 IQR & OUTLIERS:")
        print(f"  • Q1 (25th percentile): {analysis.iqr[0]:.2f}")
        print(f"  • Q3 (75th percentile): {analysis.iqr[1]:.2f}")
        print(f"  • IQR: {analysis.iqr[1] - analysis.iqr[0]:.2f}")
        print(f"  • Outliers: {len(analysis.outliers)} episodes")
        
        if analysis.outliers:
            print(f"  • Outlier values: {[round(x, 1) for x in sorted(analysis.outliers)[:5]]}...")
        
        print(f"\n🔄 TREND ANALYSIS:")
        print(f"  • Trend: {analysis.trend:.2f} loss per episode")
        if analysis.trend < -5:
            print(f"  • ✅ Model is improving (loss decreasing)")
        elif analysis.trend > 5:
            print(f"  • ❌ Model is getting worse (loss increasing)")
        else:
            print(f"  • ➡️  Stable trend")
        
        print(f"\n🎯 STABILITY METRICS:")
        print(f"  • Stability score: {analysis.stability_score:.1f}/100")
        
        if analysis.stability_score >= 80:
            print(f"  • ✅ VERY STABLE - Optimal model size")
        elif analysis.stability_score >= 60:
            print(f"  • ✅ STABLE - Model working well")
        elif analysis.stability_score >= 40:
            print(f"  • ⚠️ MODERATE - Model needs adjustment")
        else:
            print(f"  • ❌ UNSTABLE - Model needs significant changes")
        
        print(f"\n📋 RECOMMENDATIONS:")
        print(analysis.recommendation)
        
        print("\n" + "="*80)
    
    def run_analysis(self, log_path: str = None, data_str: str = None) -> VariabilityAnalysis:
        """Run complete variability analysis"""
        if log_path:
            losses = self.extract_losses_from_log(log_path)
        elif data_str:
            losses = self.parse_losses_from_data(data_str)
        else:
            # Use sample data for demonstration
            print("Using sample data for demonstration...")
            losses = [301.65, 236.62, 245.46, 582.25, 572.32, 571.88, 131.72, 244.85, 
                     472.19, 308.53, 379.42, 281.37, 502.94, 282.71, 474.05, 344.34,
                     283.42, 395.56, 119.20, 103.87, 189.52, 189.55, 374.58, 97.45,
                     653.14, 11.89, 280.66, 520.68, 276.75, 123.29, 278.19, 458.29,
                     103.09, 276.40, 187.41, 110.01, 465.97, 476.09, 365.88, 1.99,
                     566.56, 456.73, 183.66, 425.44, 218.65, 368.39, 452.83, 394.15,
                     661.30, 449.92, 202.92, 362.69, 537.49, 212.78, 207.50, 393.24,
                     916.88, 271.44, 236.08, 360.60]
        
        if not losses:
            print("No loss data available for analysis")
            return VariabilityAnalysis([], 0, 0, 0, (0, 0), (0, 0), [], 0, 0, "No data")
        
        analysis = self.analyze_variability(losses)
        self.print_detailed_analysis(analysis)
        return analysis


def main():
    parser = argparse.ArgumentParser(description='Analyze pretraining loss variability and provide model sizing recommendations')
    parser.add_argument('--log', type=str, help='Path to training log file')
    parser.add_argument('--data', type=str, help='Comma-separated loss values')
    parser.add_argument('--episodes', type=int, default=60, help='Number of episodes to analyze')
    
    args = parser.parse_args()
    
    feedback_loop = VariabilityFeedbackLoop()
    
    if args.log:
        analysis = feedback_loop.run_analysis(log_path=args.log)
    elif args.data:
        analysis = feedback_loop.run_analysis(data_str=args.data)
    else:
        # Analyze current training log
        default_log = "train_pretrain_stochastic_continue.log"
        if Path(default_log).exists():
            print(f"Analyzing default log: {default_log}")
            analysis = feedback_loop.run_analysis(log_path=default_log)
        else:
            analysis = feedback_loop.run_analysis()  # Use sample data
    
    return analysis


if __name__ == "__main__":
    main()