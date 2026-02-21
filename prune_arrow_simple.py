#!/usr/bin/env python3
"""
Simple Arrow Files Pruner - A standalone utility to prune arrow files directory

This script scans hrm/data/arrow/ directory for .feather files and keeps only
Binance-compatible files, removing or backing up others.
"""

import os
import sys
import shutil
import argparse
import re
from pathlib import Path
from datetime import datetime

# Binance basic tradepairs (USDT pairs)
BINANCE_BASIC_TRADEPAIRS = {
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", 
    "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "MATICUSDT", "LINKUSDT", "UNIUSDT",
    "ATOMUSDT", "LTCUSDT", "BCHUSDT", "ETCUSDT", "FILUSDT", "APTUSDT", 
    "OPUSDT", "ARBUSDT"
}

def format_bytes(size_bytes: int) -> str:
    """Convert bytes to human readable format."""
    if size_bytes == 0:
        return "0 B"
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"

def is_binance_format(filename: str) -> bool:
    """Check if filename follows Binance format."""
    basename = filename.replace('.feather', '')
    
    # Check basic tradepairs
    if basename in BINANCE_BASIC_TRADEPAIRS:
        return True
    
    # Check USDT format (Binance primary)
    if '_USDT' in basename:
        return True
    
    # Check BTC format (Binance secondary)
    if '_BTC' in basename:
        return True
    
    # Check other Binance formats (BNB, ETH, BUSD, EUR, AUD, GBP)
    if any(f'_{currency}' in basename for currency in ['BNB', 'ETH', 'BUSD', 'EUR', 'AUD', 'GBP']):
        return True
    
    # USD format (non-Binance - Coinbase style)
    if '_USD' in basename:
        return False
    
    # Check for simple underscore format (should be Binance)
    if '_' in basename and '-' not in basename:
        return True
    
    return False

def is_coinbase_format(filename: str) -> bool:
    """Check if filename follows Coinbase format (hyphen)."""
    basename = filename.replace('.feather', '')
    # Coinbase uses hyphen format like BTC-USD, ETH-USD
    if '-' in basename and '_' not in basename:
        return True
    return False

def scan_directory(directory: Path) -> list:
    """Scan directory for feather files."""
    feather_files = []
    
    if not directory.exists():
        print(f"Error: Directory {directory} does not exist.")
        return feather_files
    
    for file_path in directory.glob("*.feather"):
        try:
            size = file_path.stat().st_size
            feather_files.append({
                'path': file_path,
                'name': file_path.name,
                'size': size,
                'size_human': format_bytes(size),
                'is_binance': is_binance_format(file_path.name),
                'is_coinbase': is_coinbase_format(file_path.name)
            })
        except Exception as e:
            print(f"Warning: Could not read {file_path.name}: {e}")
    
    return feather_files

def categorize_files(files: list) -> dict:
    """Categorize files into groups."""
    categories = {
        'basic_tradepairs': [],
        'binance_usdt': [],
        'binance_btc': [],
        'binance_other': [],
        'coinbase': [],
        'unknown': []
    }
    
    for file_info in files:
        basename = file_info['name'].replace('.feather', '')
        
        if basename in BINANCE_BASIC_TRADEPAIRS:
            categories['basic_tradepairs'].append(file_info)
        elif file_info['is_binance']:
            if '_USDT' in basename:
                categories['binance_usdt'].append(file_info)
            elif '_BTC' in basename:
                categories['binance_btc'].append(file_info)
            else:
                categories['binance_other'].append(file_info)
        elif file_info['is_coinbase']:
            categories['coinbase'].append(file_info)
        else:
            categories['unknown'].append(file_info)
    
    return categories

def calculate_stats(categories: dict) -> dict:
    """Calculate statistics."""
    stats = {}
    
    for category, files in categories.items():
        total_size = sum(f['size'] for f in files)
        stats[category] = {
            'count': len(files),
            'total_size': total_size,
            'total_size_human': format_bytes(total_size),
            'files': files
        }
    
    # Total
    all_size = sum(s['total_size'] for s in stats.values())
    stats['total'] = {
        'count': sum(s['count'] for s in stats.values()),
        'total_size': all_size,
        'total_size_human': format_bytes(all_size)
    }
    
    # Removable (Coinbase only)
    removable_size = stats['coinbase']['total_size']
    stats['removable'] = {
        'count': stats['coinbase']['count'],
        'total_size': removable_size,
        'total_size_human': format_bytes(removable_size)
    }
    
    return stats

def generate_report(categories: dict, stats: dict, action: str, backup_dir: Path = None):
    """Generate comprehensive report."""
    print("\n" + "="*80)
    print("ARROW FILES PRUNER - COMPREHENSIVE REPORT")
    print("="*80)
    
    # Summary
    print(f"\n📊 SUMMARY:")
    print(f"  Total files scanned: {stats['total']['count']}")
    print(f"  Total size: {stats['total']['total_size_human']}")
    print(f"  Files to keep: {stats['total']['count'] - stats['removable']['count']}")
    print(f"  Files to remove: {stats['removable']['count']}")
    print(f"  Space to save: {stats['removable']['total_size_human']}")
    
    # Basic tradepairs
    if stats['basic_tradepairs']['count'] > 0:
        print(f"\n🎯 BASIC TRADEPAIRS (Binance - KEEP):")
        print(f"  Count: {stats['basic_tradepairs']['count']}")
        print(f"  Size: {stats['basic_tradepairs']['total_size_human']}")
        print(f"  Examples:")
        for f in stats['basic_tradepairs']['files'][:5]:
            print(f"    - {f['name']} ({f['size_human']})")
        if stats['basic_tradepairs']['count'] > 5:
            print(f"    ... and {stats['basic_tradepairs']['count'] - 5} more")
    
    # Binance USDT
    if stats['binance_usdt']['count'] > 0:
        print(f"\n📈 BINANCE USDT PAIRS (Binance - KEEP):")
        print(f"  Count: {stats['binance_usdt']['count']}")
        print(f"  Size: {stats['binance_usdt']['total_size_human']}")
        print(f"  Examples:")
        for f in stats['binance_usdt']['files'][:5]:
            print(f"    - {f['name']} ({f['size_human']})")
        if stats['binance_usdt']['count'] > 5:
            print(f"    ... and {stats['binance_usdt']['count'] - 5} more")
    
    # Binance BTC
    if stats['binance_btc']['count'] > 0:
        print(f"\n₿ BINANCE BTC PAIRS (Binance - KEEP):")
        print(f"  Count: {stats['binance_btc']['count']}")
        print(f"  Size: {stats['binance_btc']['total_size_human']}")
        print(f"  Examples:")
        for f in stats['binance_btc']['files'][:5]:
            print(f"    - {f['name']} ({f['size_human']})")
        if stats['binance_btc']['count'] > 5:
            print(f"    ... and {stats['binance_btc']['count'] - 5} more")
    
    # Other Binance
    if stats['binance_other']['count'] > 0:
        print(f"\n🔮 OTHER BINANCE FORMATS (Binance - KEEP):")
        print(f"  Count: {stats['binance_other']['count']}")
        print(f"  Size: {stats['binance_other']['total_size_human']}")
        print(f"  Examples:")
        for f in stats['binance_other']['files'][:5]:
            print(f"    - {f['name']} ({f['size_human']})")
        if stats['binance_other']['count'] > 5:
            print(f"    ... and {stats['binance_other']['count'] - 5} more")
    
    # Coinbase
    if stats['coinbase']['count'] > 0:
        print(f"\n❌ COINBASE FORMAT (REMOVE):")
        print(f"  Count: {stats['coinbase']['count']}")
        print(f"  Size: {stats['coinbase']['total_size_human']}")
        print(f"  Examples:")
        for f in stats['coinbase']['files'][:5]:
            print(f"    - {f['name']} ({f['size_human']})")
        if stats['coinbase']['count'] > 5:
            print(f"    ... and {stats['coinbase']['count'] - 5} more")
    
    # Unknown
    if stats['unknown']['count'] > 0:
        print(f"\n❓ UNKNOWN FORMAT (SAFE - KEEP):")
        print(f"  Count: {stats['unknown']['count']}")
        print(f"  Size: {stats['unknown']['total_size_human']}")
        print(f"  Examples:")
        for f in stats['unknown']['files'][:5]:
            print(f"    - {f['name']} ({f['size_human']})")
        if stats['unknown']['count'] > 5:
            print(f"    ... and {stats['unknown']['count'] - 5} more")
    
    # Action
    print(f"\n🎯 ACTION:")
    print(f"  Mode: {action}")
    if action == 'backup' and backup_dir:
        print(f"  Backup directory: {backup_dir}")
    elif action == 'dry_run':
        print(f"  No files will be modified (dry run)")
    
    # Space saved
    if action in ['delete', 'backup']:
        print(f"\n💾 SPACE SAVED:")
        print(f"  Files removed/backed up: {stats['removable']['count']}")
        print(f"  Space freed: {stats['removable']['total_size_human']}")
    
    print("\n" + "="*80)

def perform_action(files: list, action: str, backup_dir: Path = None) -> dict:
    """Perform action on files."""
    results = {'successful': 0, 'failed': 0, 'errors': []}
    
    if action == 'dry_run':
        results['successful'] = len(files)
        return results
    
    for file_info in files:
        try:
            if action == 'delete':
                file_info['path'].unlink()
                results['successful'] += 1
                
            elif action == 'backup':
                if backup_dir:
                    backup_dir.mkdir(parents=True, exist_ok=True)
                    dest_path = backup_dir / file_info['name']
                    shutil.move(str(file_info['path']), str(dest_path))
                    results['successful'] += 1
                else:
                    results['failed'] += 1
                    results['errors'].append(f"No backup directory specified for {file_info['name']}")
            
            elif action == 'copy':
                if backup_dir:
                    backup_dir.mkdir(parents=True, exist_ok=True)
                    dest_path = backup_dir / file_info['name']
                    shutil.copy2(str(file_info['path']), str(dest_path))
                    results['successful'] += 1
                else:
                    results['failed'] += 1
                    results['errors'].append(f"No backup directory specified for {file_info['name']}")
                    
        except Exception as e:
            results['failed'] += 1
            results['errors'].append(f"Error processing {file_info['name']}: {str(e)}")
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description='Prune arrow files directory to keep only Binance-compatible files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --dry-run                    # Show what would be done
  %(prog)s --action delete              # Remove Coinbase files
  %(prog)s --action backup --backup-dir ./backup  # Backup Coinbase files
  %(prog)s --action copy --backup-dir ./backup    # Copy Coinbase files (keep original)
  %(prog)s --directory ./arrow          # Specify custom arrow directory
        """
    )
    
    parser.add_argument('--directory', '-d', default='hrm/data/arrow',
                       help='Arrow files directory (default: hrm/data/arrow)')
    parser.add_argument('--action', '-a', choices=['dry_run', 'delete', 'backup', 'copy'],
                       default='dry_run', help='Action to perform')
    parser.add_argument('--backup-dir', '-b', help='Backup directory (required for backup/copy)')
    parser.add_argument('--output', '-o', help='Output report file')
    
    args = parser.parse_args()
    
    if args.action in ['backup', 'copy'] and not args.backup_dir:
        parser.error("--backup-dir is required for backup/copy actions")
    
    # Determine paths
    base_dir = Path(args.directory)
    if not base_dir.is_absolute():
        base_dir = Path.cwd() / base_dir
    
    backup_dir = None
    if args.backup_dir:
        backup_dir = Path(args.backup_dir)
        if not backup_dir.is_absolute():
            backup_dir = Path.cwd() / backup_dir
    
    print(f"📁 Arrow files pruner")
    print(f"  Directory: {base_dir}")
    print(f"  Action: {args.action}")
    if backup_dir:
        print(f"  Backup: {backup_dir}")
    
    # Scan
    print(f"\n🔍 Scanning for feather files...")
    files = scan_directory(base_dir)
    
    if not files:
        print("No feather files found.")
        return
    
    print(f"Found {len(files)} feather files.")
    
    # Categorize
    categories = categorize_files(files)
    stats = calculate_stats(categories)
    
    # Report
    generate_report(categories, stats, args.action, backup_dir)
    
    # Files to remove
    files_to_remove = categories['coinbase']
    
    if len(files_to_remove) == 0:
        print("\n✅ No files to remove. All files are Binance-compatible.")
        return
    
    # Confirm
    if args.action != 'dry_run':
        print(f"\n⚠️  WARNING: About to {args.action} {len(files_to_remove)} files ({stats['removable']['total_size_human']})")
        response = input("Continue? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Operation cancelled.")
            return
    
    # Execute
    print(f"\n⚙️  Performing action: {args.action}...")
    results = perform_action(files_to_remove, args.action, backup_dir)
    
    # Results
    print(f"\n✅ Action completed:")
    print(f"  Successful: {results['successful']}")
    print(f"  Failed: {results['failed']}")
    
    if results['errors']:
        print(f"\n❌ Errors:")
        for error in results['errors']:
            print(f"  - {error}")
    
    # Save report
    if args.output:
        try:
            with open(args.output, 'w') as f:
                f.write(f"Arrow Files Pruner Report\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Directory: {base_dir}\n")
                f.write(f"Action: {args.action}\n")
                f.write(f"{'='*60}\n\n")
                
                f.write(f"SUMMARY:\n")
                f.write(f"Total files: {stats['total']['count']}\n")
                f.write(f"Total size: {stats['total']['total_size_human']}\n")
                f.write(f"Files kept: {stats['total']['count'] - stats['removable']['count']}\n")
                f.write(f"Files removed: {stats['removable']['count']}\n")
                f.write(f"Space saved: {stats['removable']['total_size_human']}\n\n")
                
                f.write(f"CATEGORIES:\n")
                for cat in ['basic_tradepairs', 'binance_usdt', 'binance_btc', 'binance_other', 'coinbase', 'unknown']:
                    f.write(f"\n{cat}:\n")
                    f.write(f"  Count: {stats[cat]['count']}\n")
                    f.write(f"  Size: {stats[cat]['total_size_human']}\n")
            
            print(f"\n📄 Report saved to: {args.output}")
        except Exception as e:
            print(f"Error saving report: {e}")

if __name__ == "__main__":
    main()