#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
echo "============================================================================"
echo "  中国象棋 后训练强化学习探索 (DeepSeek-R1 / DAPO GRPO CoT Post-Training)"
echo "============================================================================"
python3 train_rl_grpo.py "$@"
