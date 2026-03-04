#!/usr/bin/env bash
set -euo pipefail

# Canonical launcher for the active MoneyFan trainer + web console stack.
# Defaults come from trainerd.py (port 8080, etc.); any args are forwarded.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

exec python3 trainerd.py "$@"
