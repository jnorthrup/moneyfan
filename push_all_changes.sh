#!/bin/bash
# push_all_changes.sh
# Push all changes to main branch
# This will make the live paper trading system visible to everyone

set -e

echo "============================================================"
echo "MONEYFAN - PUSH ALL CHANGES TO MAIN"
echo "============================================================"
echo ""

# Check if we're in the right directory
if [ ! -f "vector_store.py" ]; then
    echo "❌ ERROR: Not in moneyfan directory"
    echo "Please run this script from /Users/jim/work/moneyfan"
    exit 1
fi

echo "✅ In moneyfan directory"

# Check git status
echo ""
echo "📊 Current git status:"
git status --short

# Add all new files
echo ""
echo "📦 Adding new files to git..."
git add execution/live_executor.py
git add mvp_runner.py
git add launch_live_paper.sh
git add monitor_paper_trading.sh
git add GOALS.md

# Check for any untracked files
echo ""
echo "🔍 Checking for untracked files..."
UNTRACKED=$(git ls-files --others --exclude-standard | grep -v "paper_results" | grep -v "__pycache__" | grep -v ".pyc" | head -10)
if [ -n "$UNTRACKED" ]; then
    echo "⚠️  Untracked files found (not added):"
    echo "$UNTRACKED"
    echo ""
    echo "Add these files if you want to include them:"
    echo "git add [filename]"
else
    echo "✅ No untracked files"
fi

# Create commit message
echo ""
echo "📝 Creating commit message..."
COMMIT_MSG="feat: Launch live paper trading on Coinbase Advanced Trade

- Create live_executor.py for direct Coinbase SDK integration
- Update mvp_runner.py with live execution capability
- Add launch_live_paper.sh for easy deployment
- Add monitor_paper_trading.sh for 24h monitoring
- GOALS.md updated with 29-line high-entropy focus

System now 100% Python/MLX (no JS/Kotlin bridges)
Ready for live paper trading with real Coinbase API keys

Usage:
1. Export API keys:
   export COINBASE_API_KEY=\"your-key\"
   export COINBASE_API_SECRET=\"your-secret\"

2. Launch paper trading:
   ./launch_live_paper.sh

3. Monitor in separate terminal:
   ./monitor_paper_trading.sh

Expected output:
- 51 ticks/sec processing rate
- Vector store operations in 0.25ms
- Direct Coinbase SDK execution
- Paper trading results in paper_results/

Next steps after launch:
1. Monitor 24 hours for baseline P&L
2. Scale to 8 predictors if profit factor > 1.5
3. Run vector cache ablation test
4. Live trading optimization"

# Check git status again
echo ""
echo "📦 Git status before commit:"
git status --short

# Show diff
echo ""
echo "🔍 Diff summary:"
git diff --stat 2>/dev/null || echo "No changes to diff"

# Confirm commit
echo ""
echo "Ready to commit and push!"
echo "Commit message:"
echo "------------------------------------------------------------"
echo "$COMMIT_MSG" | head -20
echo "------------------------------------------------------------"
echo ""
echo "Type 'yes' to commit and push, or 'no' to cancel:"
read -p "> " CONFIRM

if [ "$CONFIRM" != "yes" ]; then
    echo "❌ Commit cancelled"
    exit 0
fi

# Commit
echo ""
echo "📝 Committing changes..."
git commit -m "$COMMIT_MSG"

# Push
echo ""
echo "🚀 Pushing to remote..."
git push

echo ""
echo "============================================================"
echo "✅ ALL CHANGES PUSHED TO MAIN"
echo "============================================================"
echo ""
echo "Next steps:"
echo "1. Wait 2 minutes for GitHub sync"
echo "2. Verify changes: git log --oneline -3"
echo "3. Launch live paper trading:"
echo "   ./launch_live_paper.sh"
echo ""
echo "🎉 Live paper trading is now visible to everyone!"
echo "============================================================"