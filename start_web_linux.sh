#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
chmod +x engine/Pikafish-Linux-x86-64-universal 2>/dev/null || true
python3 web_app.py --port 7860
