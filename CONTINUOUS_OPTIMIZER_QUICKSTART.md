# Continuous Model Size Optimizer - Quick Start Guide

## 🎯 **SYSTEM COMPLETE - READY TO USE**

### **What Was Built:**
1. ✅ **Modified train.py** - Supports `--hidden-dim`, `--regime-layers`, `--tactical-layers`, `--attention-heads`
2. ✅ **Variability feedback loop** - Analyzes pretraining loss (CV) to determine optimal size
3. ✅ **Continuous optimizer** - Automatically tests configurations until optimal found
4. ✅ **Autograd optimizer** - Uses MLX autograd for gradient-based optimization (optional)

## 🚀 **QUICK START**

### **Option 1: Quick Automatic Optimization**
```bash
# Test 3 configurations quickly (15 episodes each)
python continuous_optimizer.py --quick --max-tests 3
```

### **Option 2: Manual Testing**
```bash
# Test a specific configuration
python train.py --episodes 20 --notional 100 --pretrain-only --fully-stochastic-pair-sampling --hidden-dim 96 --regime-layers 3 --tactical-layers 3 --attention-heads 6

# Quick analysis
python quick_variability_check.py
```

### **Option 3: Full Optimization (Recommended)**
```bash
# Run continuous optimization until optimal size found
python continuous_optimizer.py --episodes 30 --max-tests 8
```

## 📊 **UNDERSTANDING THE RESULTS**

### **Key Metrics:**
- **CV (Coefficient of Variation)**: `std/mean` - Lower is better (target: <0.4)
- **Mean Loss**: Average prediction error (target: <200)
- **Stability Score**: 0-100 (target: >60)

### **Configuration Evaluation:**
```
CV < 0.2:    Model may be overpowered/overfitting
CV 0.2-0.4:  ✅ OPTIMAL - Good model sizing
CV 0.4-0.6:  ⚠️ BORDERLINE - Model underpowered
CV > 0.6:    ❌ UNDERPOWERED - Need larger model
```

## 🔧 **FILES CREATED**

| File | Purpose |
|------|---------|
| `continuous_optimizer.py` | Main optimizer - finds optimal size automatically |
| `autograd_model_optimizer.py` | MLX autograd-based optimization |
| `quick_optimizer.py` | Quick gradient-free optimization |
| `quick_variability_check.py` | 1-line variability analysis |
| `variability_feedback_loop.py` | Comprehensive analysis |
| `model_sizing_optimizer.py` | Advanced recommendations |
| `test_new_config.py` | Test new arguments |
| `continuous_optimizer_daemon.py` | Long-running optimization daemon |

## 📋 **CURRENT STATUS**

### **Stochastic Pretraining (Original):**
- ✅ **Running**: 96/1000 episodes (9.6% complete)
- **PID**: 72211
- **Status**: Continue if you want complete data

### **Model Sizing Assessment:**
- **Current model**: `hidden_dim=64` (underpowered)
- **Current CV**: 0.574 (too high)
- **Target CV**: <0.4 (optimal)
- **Recommendation**: Test `hidden_dim=96-128`

## 💡 **RECOMMENDED NEXT STEPS**

### **Step 1: Quick Test (30 minutes)**
```bash
# Test 3 configurations
python continuous_optimizer.py --quick --max-tests 3
```

### **Step 2: Analyze Results**
```bash
# Check variability of any test
python quick_variability_check.py
```

### **Step 3: Full Training**
```bash
# Once optimal config found, run full training
python train.py --episodes 1000 --notional 100 --pretrain-only --fully-stochastic-pair-sampling --hidden-dim 128 --regime-layers 3 --tactical-layers 3 --attention-heads 6
```

## 🎯 **AUTOGRAID OPTIMIZATION (Advanced)**

If MLX is installed, use autograd for gradient-based optimization:

```bash
python autograd_model_optimizer.py --quick --iterations 5
```

**Note**: Autograd requires MLX (Apple Silicon). If not available, use gradient-free optimizer.

## 📈 **EXPECTED TIMING**

- **Quick test (3 configs × 15 episodes)**: ~45 minutes
- **Full optimization (8 configs × 30 episodes)**: ~4 hours
- **Full training (1000 episodes)**: ~2 hours (with optimal config)

## ⚠️ **IMPORTANT NOTES**

1. **Current training in background**: Will continue unless stopped
2. **New configs train from scratch**: No weight loading for different sizes
3. **Results vary**: Run each config 2-3 times for confidence
4. **Optimal CV range**: 0.2-0.4 (lower than current 0.574)

## 🔍 **VERIFICATION COMMANDS**

```bash
# Check if system is ready
python quick_variability_check.py

# Test new arguments work
python train.py --episodes 5 --pretrain-only --hidden-dim 96 2>&1 | head -10

# Monitor running training
tail -f train_pretrain_stochastic_continue.log

# Stop all training
pkill -f "train.py"
```

## ✅ **SYSTEM STATUS: READY**

The continuous optimization system is fully operational and ready to find the optimal model size for your stochastic pretraining task! 🚀