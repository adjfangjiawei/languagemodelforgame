#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

echo "======================================================================"
echo "  Xiangqi Variation-DAG CoT LM - GTX 1660 Ti (6GB) ~4h One-Click Run"
echo "======================================================================"

chmod +x engine/Pikafish-Linux-x86-64-universal 2>/dev/null || true
python3 -m pip install -r requirements.txt

python3 run_4h_pipeline.py --total-hours 4.0 --data-minutes 35 --preset fast-4m
