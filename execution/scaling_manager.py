"""
Scaling Manager - Scale from 3 to 8 Predictors
==============================================

Handles scaling from MVP (3 predictors) to full system (8 predictors).
Includes ablation testing for vector cache on/off.

Features:
- Dynamic predictor configuration
- Ablation testing framework
- Performance comparison
- Gradual scaling with validation
"""

import asyncio
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from test_time_predictor import create_short_horizon_predictor
from vector_store import create_simple_vector_store

class ScalingStage(Enum):
    """Scaling stages"""
    MVP = "mvp"           # 3 predictors
    SCALED = "scaled"     # 8 predictors
    ABLATION = "ablation" # Vector cache on/off test

@dataclass
class ScalingConfig:
    """Configuration for scaling"""
    mvp_predictors: int = 3
    scaled_predictors: int = 8
    predictor_types: List[str] = field(default_factory=lambda: [
        "transformer_5m",
        "xgboost_15m", 
        "lightgbm_1h",
        "transformer_30m",
        "xgboost_1h",
        "lightgbm_4h",
        "transformer_1d",
        "xgboost_1d"
    ])
    
    # Ablation settings
    ablation_runs: int = 2  # Run each config twice
    run_duration_hours: int = 4  # Hours per run
    
    # Performance thresholds
    min_profit_factor: float = 1.5
    min_sharpe: float = 1.0
    max_drawdown: float = 0.15
    
    # Scaling rules
    scale_on_profit_factor: float = 2.0  # Scale if PF > 2.0
    scale_on_sharpe: float = 1.5  # Scale if Sharpe > 1.5

class ScalingManager:
    """
    Manage scaling from 3 to 8 predictors with ablation testing
    """
    
    def __init__(self, config: ScalingConfig):
        self.config = config
        self.current_stage = ScalingStage.MVP
        self.predictors = []
        self.vector_store = None
        self.vector_cache_enabled = True
        
        # Performance tracking
        self.performance_history: Dict[str, List[Dict[str, Any]]] = {
            "mvp": [],
            "scaled": [],
            "ablation_on": [],
            "ablation_off": []
        }
        
        print(f"[ScalingManager] Initialized with config: {config}")
    
    def setup_predictors(self, num_predictors: int) -> List[Any]:
        """Setup predictors for scaling stage"""
        predictors = []
        
        for i in range(num_predictors):
            predictor = create_short_horizon_predictor()
            
            # Set model paths based on predictor type
            if i < len(self.config.predictor_types):
                ptype = self.config.predictor_types[i]
                if "transformer" in ptype:
                    predictor.config.model_paths = [f"models/{ptype}.mlxbf"]
                elif "xgboost" in ptype:
                    predictor.config.model_paths = [f"models/{ptype}.mlxbf"]
                elif "lightgbm" in ptype:
                    predictor.config.model_paths = [f"models/{ptype}.mlxbf"]
            
            predictors.append(predictor)
        
        print(f"[ScalingManager] Setup {num_predictors} predictors")
        return predictors
    
    async def run_mvp(self, data_stream, duration_hours: int = 4) -> Dict[str, Any]:
        """Run MVP stage (3 predictors)"""
        print(f"\n{'='*60}")
        print("SCALING: MVP STAGE (3 predictors)")
        print(f"{'='*60}")
        
        self.current_stage = ScalingStage.MVP
        self.predictors = self.setup_predictors(self.config.mvp_predictors)
        self.vector_store = create_simple_vector_store(vector_dim=64)
        self.vector_cache_enabled = True
        
        # Run simulation
        results = await self._run_simulation(
            data_stream=data_stream,
            duration_hours=duration_hours,
            stage_name="mvp"
        )
        
        # Store results
        self.performance_history["mvp"].append(results)
        
        return results
    
    async def run_scaled(self, data_stream, duration_hours: int = 4) -> Dict[str, Any]:
        """Run scaled stage (8 predictors)"""
        print(f"\n{'='*60}")
        print("SCALING: SCALED STAGE (8 predictors)")
        print(f"{'='*60}")
        
        self.current_stage = ScalingStage.SCALED
        self.predictors = self.setup_predictors(self.config.scaled_predictors)
        self.vector_store = create_simple_vector_store(vector_dim=64)
        self.vector_cache_enabled = True
        
        # Run simulation
        results = await self._run_simulation(
            data_stream=data_stream,
            duration_hours=duration_hours,
            stage_name="scaled"
        )
        
        # Store results
        self.performance_history["scaled"].append(results)
        
        return results
    
    async def run_ablation(self, data_stream, duration_hours: int = 2) -> Dict[str, Any]:
        """Run ablation test (vector cache on vs off)"""
        print(f"\n{'='*60}")
        print("SCALING: ABLATION TEST (Vector cache ON vs OFF)")
        print(f"{'='*60}")
        
        # Run with vector cache ON
        print(">>> Running with vector cache ENABLED...")
        self.vector_cache_enabled = True
        self.predictors = self.setup_predictors(self.config.mvp_predictors)  # Use 3 for fair comparison
        self.vector_store = create_simple_vector_store(vector_dim=64)
        
        results_on = await self._run_simulation(
            data_stream=data_stream,
            duration_hours=duration_hours,
            stage_name="ablation_on"
        )
        self.performance_history["ablation_on"].append(results_on)
        
        # Run with vector cache OFF
        print("\n>>> Running with vector cache DISABLED...")
        self.vector_cache_enabled = False
        self.vector_store = None  # No vector cache
        
        results_off = await self._run_simulation(
            data_stream=data_stream,
            duration_hours=duration_hours,
            stage_name="ablation_off"
        )
        self.performance_history["ablation_off"].append(results_off)
        
        # Compare results
        comparison = {
            "with_cache": results_on,
            "without_cache": results_off,
            "delta_profit_factor": results_on.get("profit_factor", 0) - results_off.get("profit_factor", 0),
            "delta_sharpe": results_on.get("sharpe", 0) - results_off.get("sharpe", 0),
            "delta_max_drawdown": results_on.get("max_drawdown", 0) - results_off.get("max_drawdown", 0)
        }
        
        print(f"\n{'='*60}")
        print("ABLATION COMPARISON")
        print(f"{'='*60}")
        print(f"Profit Factor - Cache ON: {results_on.get('profit_factor', 0):.2f}")
        print(f"Profit Factor - Cache OFF: {results_off.get('profit_factor', 0):.2f}")
        print(f"Delta: {comparison['delta_profit_factor']:+.2f}")
        print(f"Sharpe - Cache ON: {results_on.get('sharpe', 0):.2f}")
        print(f"Sharpe - Cache OFF: {results_off.get('sharpe', 0):.2f}")
        print(f"Delta: {comparison['delta_sharpe']:+.2f}")
        
        return comparison
    
    async def _run_simulation(self, data_stream, duration_hours: int, stage_name: str) -> Dict[str, Any]:
        """Run simulation for given stage"""
        # This would integrate with mvp_runner.py
        # For now, return simulated results
        await asyncio.sleep(1)  # Simulate some work
        
        return {
            "stage": stage_name,
            "profit_factor": np.random.uniform(1.2, 2.5),
            "sharpe": np.random.uniform(0.8, 2.0),
            "max_drawdown": np.random.uniform(0.05, 0.18),
            "win_rate": np.random.uniform(0.4, 0.6),
            "total_trades": np.random.randint(50, 200),
            "predictors": len(self.predictors),
            "vector_cache_enabled": self.vector_cache_enabled
        }
    
    def check_scaling_criteria(self, results: Dict[str, Any]) -> Tuple[bool, str]:
        """Check if scaling criteria are met"""
        pf = results.get("profit_factor", 0)
        sharpe = results.get("sharpe", 0)
        
        if pf >= self.config.scale_on_profit_factor and sharpe >= self.config.scale_on_sharpe:
            return True, f"PF={pf:.2f} >= {self.config.scale_on_profit_factor}, Sharpe={sharpe:.2f} >= {self.config.scale_on_sharpe}"
        else:
            return False, f"PF={pf:.2f} or Sharpe={sharpe:.2f} below thresholds"
    
    def generate_report(self) -> str:
        """Generate scaling report"""
        report = []
        report.append("="*80)
        report.append("SCALING MANAGER - COMPREHENSIVE REPORT")
        report.append("="*80)
        report.append("")
        
        # MVP results
        if self.performance_history["mvp"]:
            mvp_results = self.performance_history["mvp"][-1]
            report.append("MVP STAGE (3 Predictors):")
            report.append(f"  Profit Factor: {mvp_results.get('profit_factor', 0):.2f}")
            report.append(f"  Sharpe Ratio: {mvp_results.get('sharpe', 0):.2f}")
            report.append(f"  Max Drawdown: {mvp_results.get('max_drawdown', 0):.2%}")
            report.append(f"  Win Rate: {mvp_results.get('win_rate', 0):.1f}%")
            report.append(f"  Total Trades: {mvp_results.get('total_trades', 0)}")
            report.append("")
        
        # Scaled results
        if self.performance_history["scaled"]:
            scaled_results = self.performance_history["scaled"][-1]
            report.append("SCALED STAGE (8 Predictors):")
            report.append(f"  Profit Factor: {scaled_results.get('profit_factor', 0):.2f}")
            report.append(f"  Sharpe Ratio: {scaled_results.get('sharpe', 0):.2f}")
            report.append(f"  Max Drawdown: {scaled_results.get('max_drawdown', 0):.2%}")
            report.append(f"  Win Rate: {scaled_results.get('win_rate', 0):.1f}%")
            report.append(f"  Total Trades: {scaled_results.get('total_trades', 0)}")
            report.append("")
        
        # Ablation results
        if self.performance_history["ablation_on"] and self.performance_history["ablation_off"]:
            ablation_on = self.performance_history["ablation_on"][-1]
            ablation_off = self.performance_history["ablation_off"][-1]
            
            report.append("ABLATION TEST (Vector Cache ON vs OFF):")
            report.append(f"  WITH CACHE:")
            report.append(f"    Profit Factor: {ablation_on.get('profit_factor', 0):.2f}")
            report.append(f"    Sharpe Ratio: {ablation_on.get('sharpe', 0):.2f}")
            report.append(f"  WITHOUT CACHE:")
            report.append(f"    Profit Factor: {ablation_off.get('profit_factor', 0):.2f}")
            report.append(f"    Sharpe Ratio: {ablation_off.get('sharpe', 0):.2f}")
            report.append(f"  Delta PF: {ablation_on.get('profit_factor', 0) - ablation_off.get('profit_factor', 0):+.2f}")
            report.append(f"  Delta Sharpe: {ablation_on.get('sharpe', 0) - ablation_off.get('sharpe', 0):+.2f}")
            report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS:")
        if self.performance_history["mvp"]:
            mvp_ok = self.check_scaling_criteria(self.performance_history["mvp"][-1])
            if mvp_ok[0]:
                report.append("  ✅ MVP stage passed scaling criteria - ready to scale to 8 predictors")
            else:
                report.append(f"  ❌ MVP stage needs improvement: {mvp_ok[1]}")
        
        if self.performance_history["ablation_on"] and self.performance_history["ablation_off"]:
            cache_on_better = (
                self.performance_history["ablation_on"][-1].get("profit_factor", 0) >
                self.performance_history["ablation_off"][-1].get("profit_factor", 0)
            )
            if cache_on_better:
                report.append("  ✅ Vector cache ON improves performance - keep enabled")
            else:
                report.append("  ❌ Vector cache OFF performs better - consider disabling")
        
        return "\n".join(report)

# Example usage
async def main():
    config = ScalingConfig()
    manager = ScalingManager(config)
    
    # Simulate data stream
    async def mock_data_stream():
        for i in range(100):
            yield {"timestamp": time.time(), "price": 50000 + i, "volume": 1000}
            await asyncio.sleep(0.01)
    
    # Run MVP stage
    mvp_results = await manager.run_mvp(mock_data_stream(), duration_hours=1)
    print(f"MVP results: {mvp_results}")
    
    # Check scaling criteria
    can_scale, reason = manager.check_scaling_criteria(mvp_results)
    if can_scale:
        print(f"\n✅ Can scale to 8 predictors: {reason}")
        
        # Run scaled stage
        scaled_results = await manager.run_scaled(mock_data_stream(), duration_hours=1)
        print(f"Scaled results: {scaled_results}")
    else:
        print(f"\n❌ Cannot scale yet: {reason}")
    
    # Run ablation test
    print("\n" + "="*60)
    print("RUNNING ABLATION TEST")
    print("="*60)
    ablation_results = await manager.run_ablation(mock_data_stream(), duration_hours=1)
    
    # Generate report
    report = manager.generate_report()
    print("\n" + report)
    
    # Save report
    report_path = Path("scaling_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")

if __name__ == "__main__":
    asyncio.run(main())