#!/bin/bash

# Arrow Files Pruner Shell Script
# A simple interface to prune arrow files directory

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/prune_arrow_simple.py"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed or not in PATH"
    exit 1
fi

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script not found at $PYTHON_SCRIPT"
    exit 1
fi

# Help function
show_help() {
    cat << EOF
Arrow Files Pruner - Prune arrow files to keep only Binance-compatible files

Usage: $0 [OPTIONS]

Options:
  -d, --directory DIR    Arrow files directory (default: hrm/data/arrow)
  -a, --action ACTION    Action: dry_run, delete, backup, copy (default: dry_run)
  -b, --backup-dir DIR   Backup directory (required for backup/copy)
  -o, --output FILE      Output report file
  -h, --help             Show this help message

Examples:
  $0 --dry-run                    # Show what would be done
  $0 --action delete              # Remove Coinbase files
  $0 --action backup --backup-dir ./backup  # Backup Coinbase files
  $0 --action copy --backup-dir ./backup    # Copy Coinbase files (keep original)
  $0 --directory ./arrow          # Specify custom arrow directory

Actions:
  dry_run    - Show report without modifying files
  delete     - Remove non-Binance (Coinbase) files
  backup     - Move non-Binance files to backup directory
  copy       - Copy non-Binance files to backup directory (keep original)

EOF
}

# Parse arguments
ACTION="dry_run"
DIRECTORY="hrm/data/arrow"
BACKUP_DIR=""
OUTPUT=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--directory)
            DIRECTORY="$2"
            shift 2
            ;;
        -a|--action)
            ACTION="$2"
            shift 2
            ;;
        -b|--backup-dir)
            BACKUP_DIR="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Build command
CMD="python3 \"$PYTHON_SCRIPT\" --action \"$ACTION\" --directory \"$DIRECTORY\""

if [ -n "$BACKUP_DIR" ]; then
    CMD="$CMD --backup-dir \"$BACKUP_DIR\""
fi

if [ -n "$OUTPUT" ]; then
    CMD="$CMD --output \"$OUTPUT\""
fi

echo "Running: $CMD"
echo ""

# Execute
eval $CMD