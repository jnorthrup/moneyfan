#!/usr/bin/env python3
"""
Model Sizing Optimizer with Feedback Loop
==========================================

Advanced variability analysis with recommendations for optimal model sizing.
This script analyzes pretraining loss patterns and provides specific
recommendations for model configuration adjustments.

Usage:
    python model_sizing_optimizer.py --log train_pretrain_stochastic_continue.log
    python model_sizing_optimizer.py --analyze current --compare-with previous
"""

import re
import numpy as np
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import json


@dataclass
class ModelConfig:
    """Model configuration parameters"""
    hidden_dim: int = 64
    regime_layers: int = 2
    tactical_layers: int = 2
    attention_heads: int = 4
    codec_outputs: int = 24
    
    def to_string(self) -> str:
        return f"hidden={self.hidden_dim}, layers=regime:{self.regime_layers}/tactical:{self.tactical_layers}, heads={self.attention_heads}, codecs={self.codec_outputs}"


@dataclass
class SizingRecommendation:
    """Specific recommendation for model sizing"""
    config: ModelConfig
    confidence: float  # 0-1
    expected_improvement: str
    reasoning: str
    priority: str  # high, medium, low


class ModelSizingOptimizer:
    """Optimizes model sizing based on pretraining loss variability"""
    
    def __init__(self):
        # Current model configuration
        self.current_config = ModelConfig(
            hidden_dim=64,
            regime_layers=2,
            tactical_layers=2,
            attention_heads=4,
            codec_outputs=24
        )
        
        # Model sizing guidelines based on input dimension
        self.sizing_guidelines = {
            # For input_dim 104 (current)
            104: {
                'underpowered': {'hidden_dim': 64, 'stability_score': 0, 'loss_mean': 400},
                'borderline': {'hidden_dim': 128, 'stability_score': 50, 'loss_mean': 200},
                'optimal': {'hidden_dim': 256, 'stability_score': 70, 'loss_mean': 100},
                'overpowered': {'hidden_dim': 512, 'stability_score': 80, 'loss_mean': 50}
            }
        }
    
    def analyze_loss_patterns(self, losses: List[float]) -> Dict:
        """Analyze loss patterns to understand model performance"""
        if not losses:
            return {}
        
        losses_array = np.array(losses)
        
        analysis = {
            'mean': float(np.mean(losses_array)),
            'std': float(np.std(losses_array)),
            'min': float(np.min(losses_array)),
            'max': float(np.max(losses_array)),
            'median': float(np.median(losses_array)),
            'cv': float(np.std(losses_array) / np.mean(losses_array)),
            'range': float(np.max(losses_array) - np.min(losses_array)),
            'percentiles': {
                '10': float(np.percentile(losses_array, 10)),
                '25': float(np.percentile(losses_array, 25)),
                '50': float(np.percentile(losses_array, 50)),
                '75': float(np.percentile(losses_array, 75)),
                '90': float(np.percentile(losses_array, 90))
            }
        }
        
        # Determine performance category
        if analysis['mean'] > 400:
            analysis['performance'] = "underpowered"
        elif analysis['mean'] > 200:
            analysis['performance'] = "borderline"
        elif analysis['mean'] > 100:
            analysis['performance'] = "optimal"
        else:
            analysis['performance'] = "overpowered"
        
        # Variability assessment
        if analysis['cv'] > 0.5:
            analysis['variability'] = "high"
        elif analysis['cv'] > 0.3:
            analysis['variability'] = "moderate"
        else:
            analysis['variability'] = "low"
        
        return analysis
    
    def extract_losses_from_log(self, log_path: str) -> List[float]:
        """Extract pretraining losses from log file"""
        try:
            with open(log_path, 'r') as f:
                content = f.read()
            
            pattern = r'pred_loss=([\d.]+)'
            matches = re.findall(pattern, content)
            return [float(match) for match in matches]
        except Exception as e:
            print(f"Error reading log: {e}")
            return []
    
    def generate_config_recommendations(self, analysis: Dict) -> List[SizingRecommendation]:
        """Generate specific configuration recommendations"""
        recommendations = []
        
        # Base recommendation
        if analysis['performance'] == "underpowered":
            # Current model is too small
            config1 = ModelConfig(
                hidden_dim=128,  # Double the current
                regime_layers=3,  # Add one more layer
                tactical_layers=3,
                attention_heads=8,
                codec_outputs=24
            )
            recommendations.append(SizingRecommendation(
                config=config1,
                confidence=0.8,
                expected_improvement="Expected 30-50% reduction in loss variability",
                reasoning=f"Current mean loss {analysis['mean']:.1f} is high. Model needs more capacity.",
                priority="high"
            ))
            
            # Even larger option
            config2 = ModelConfig(
                hidden_dim=256,
                regime_layers=4,
                tactical_layers=4,
                attention_heads=8,
                codec_outputs=32
            )
            recommendations.append(SizingRecommendation(
                config=config2,
                confidence=0.6,
                expected_improvement="Expected 50-70% reduction in loss variability",
                reasoning="If current model is severely underpowered, double capacity may be needed.",
                priority="medium"
            ))
        
        elif analysis['performance'] == "borderline":
            # Current model might need slight adjustment
            config1 = ModelConfig(
                hidden_dim=96,  # 50% increase
                regime_layers=2,
                tactical_layers=2,
                attention_heads=6,
                codec_outputs=24
            )
            recommendations.append(SizingRecommendation(
                config=config1,
                confidence=0.7,
                expected_improvement="Expected 15-25% reduction in loss variability",
                reasoning=f"Current mean loss {analysis['mean']:.1f} suggests model is borderline.",
                priority="medium"
            ))
            
            # Alternative: Add layers instead of width
            config2 = ModelConfig(
                hidden_dim=64,
                regime_layers=3,
                tactical_layers=3,
                attention_heads=6,
                codec_outputs=24
            )
            recommendations.append(SizingRecommendation(
                config=config2,
                confidence=0.6,
                expected_improvement="Expected 10-20% reduction in loss variability",
                reasoning="Adding layers instead of width for better feature extraction.",
                priority="medium"
            ))
        
        elif analysis['performance'] == "optimal":
            # Current model might be overpowered
            config1 = ModelConfig(
                hidden_dim=48,  # 25% smaller
                regime_layers=2,
                tactical_layers=2,
                attention_heads=4,
                codec_outputs=24
            )
            recommendations.append(SizingRecommendation(
                config=config1,
                confidence=0.4,
                expected_improvement="Expected 10-20% faster training with similar performance",
                reasoning=f"Current mean loss {analysis['mean']:.1f} is already low. Can try smaller model.",
                priority="low"
            ))
        
        # Special recommendation based on variability
        if analysis['variability'] == "high":
            # Add recommendation for reducing variability
            config_var = ModelConfig(
                hidden_dim=self.current_config.hidden_dim,
                regime_layers=self.current_config.regime_layers + 1,
                tactical_layers=self.current_config.tactical_layers + 1,
                attention_heads=self.current_config.attention_heads + 4,
                codec_outputs=self.current_config.codec_outputs
            )
            recommendations.append(SizingRecommendation(
                config=config_var,
                confidence=0.7,
                expected_improvement="Expected 20-30% reduction in loss variability",
                reasoning=f"High CV ({analysis['cv']:.2f}) indicates model struggles with diverse patterns. More layers may help.",
                priority="high"
            ))
        
        return recommendations
    
    def extract_baseline_data(self) -> Optional[Dict]:
        """Extract baseline data from previous training runs"""
        baseline_file = "train_pretrain_stochastic_1000.log"
        if Path(baseline_file).exists():
            baseline_losses = self.extract_losses_from_log(baseline_file)
            if baseline_losses:
                return self.analyze_loss_patterns(baseline_losses)
        return None
    
    def compare_with_baseline(self, current_analysis: Dict, baseline_analysis: Dict) -> str:
        """Compare current analysis with baseline"""
        if not baseline_analysis:
            return "No baseline data available for comparison."
        
        comparison = []
        comparison.append("📊 BASELINE COMPARISON:")
        comparison.append(f"  • Current mean loss: {current_analysis['mean']:.1f} vs Baseline: {baseline_analysis['mean']:.1f}")
        comparison.append(f"  • Current CV: {current_analysis['cv']:.3f} vs Baseline: {baseline_analysis['cv']:.3f}")
        comparison.append(f"  • Current max loss: {current_analysis['max']:.1f} vs Baseline: {baseline_analysis['max']:.1f}")
        
        # Improvement analysis
        mean_improvement = ((baseline_analysis['mean'] - current_analysis['mean']) / baseline_analysis['mean']) * 100
        cv_improvement = ((baseline_analysis['cv'] - current_analysis['cv']) / baseline_analysis['cv']) * 100
        
        if mean_improvement > 10:
            comparison.append(f"  • ✅ Mean loss improved by {mean_improvement:.1f}%")
        elif mean_improvement < -10:
            comparison.append(f"  • ❌ Mean loss worsened by {abs(mean_improvement):.1f}%")
        
        if cv_improvement > 10:
            comparison.append(f"  • ✅ Variability improved by {cv_improvement:.1f}%")
        elif cv_improvement < -10:
            comparison.append(f"  • ❌ Variability worsened by {abs(cv_improvement):.1f}%")
        
        return "\n".join(comparison)
    
    def print_recommendations(self, recommendations: List[SizingRecommendation], analysis: Dict, baseline_comparison: str = ""):
        """Print formatted recommendations"""
        print("\n" + "="*80)
        print("MODEL SIZING OPTIMIZER - FEEDBACK LOOP ANALYSIS")
        print("="*80)
        
        print(f"\n📈 CURRENT MODEL PERFORMANCE:")
        print(f"  • Current config: {self.current_config.to_string()}")
        print(f"  • Performance: {analysis['performance'].upper()}")
        print(f"  • Variability: {analysis['variability'].upper()}")
        print(f"  • Mean loss: {analysis['mean']:.1f}")
        print(f"  • Loss range: {analysis['min']:.1f} - {analysis['max']:.1f}")
        print(f"  • CV: {analysis['cv']:.3f}")
        
        if baseline_comparison:
            print(f"\n{baseline_comparison}")
        
        print(f"\n🎯 RECOMMENDATIONS:")
        
        if not recommendations:
            print("  • ✅ Current model appears well-sized")
            return
        
        for i, rec in enumerate(recommendations, 1):
            print(f"\n  {i}. {rec.priority.upper()} PRIORITY (Confidence: {rec.confidence:.0%})")
            print(f"     Config: {rec.config.to_string()}")
            print(f"     Expected: {rec.expected_improvement}")
            print(f"     Reasoning: {rec.reasoning}")
        
        print(f"\n💡 IMPLEMENTATION GUIDE:")
        print("  1. Start with highest priority recommendation")
        print("  2. Run small test (100-200 episodes) with new config")
        print("  3. Compare loss variability with current training")
        print("  4. If improvement > 15%, proceed with full training")
        print("  5. If no improvement, try next recommendation")
        
        print("\n" + "="*80)
    
    def run_optimization(self, log_path: str = None) -> Dict:
        """Run complete optimization analysis"""
        
        # Get current training data
        if log_path and Path(log_path).exists():
            losses = self.extract_losses_from_log(log_path)
        elif Path("train_pretrain_stochastic_continue.log").exists():
            losses = self.extract_losses_from_log("train_pretrain_stochastic_continue.log")
        else:
            print("No training log found. Using sample data for demonstration.")
            # Sample data representing current performance
            losses = [301.65, 236.62, 245.46, 582.25, 572.32, 571.88, 131.72, 244.85,
                     472.19, 308.53, 379.42, 281.37, 502.94, 282.71, 474.05, 344.34]
        
        if not losses:
            print("No loss data available")
            return {}
        
        # Analyze current performance
        analysis = self.analyze_loss_patterns(losses)
        
        # Compare with baseline if available
        baseline_analysis = self.extract_baseline_data()
        baseline_comparison = self.compare_with_baseline(analysis, baseline_analysis) if baseline_analysis else ""
        
        # Generate recommendations
        recommendations = self.generate_config_recommendations(analysis)
        
        # Print results
        self.print_recommendations(recommendations, analysis, baseline_comparison)
        
        return {
            'analysis': analysis,
            'recommendations': recommendations,
            'baseline_comparison': baseline_comparison
        }


def main():
    parser = argparse.ArgumentParser(description='Model sizing optimizer with feedback loop')
    parser.add_argument('--log', type=str, help='Path to training log file')
    parser.add_argument('--analyze', action='store_true', help='Analyze current training')
    
    args = parser.parse_args()
    
    optimizer = ModelSizingOptimizer()
    result = optimizer.run_optimization(args.log)
    
    return result


if __name__ == "__main__":
    main()