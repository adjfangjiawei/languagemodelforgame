#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
echo "============================================================================"
echo "  中国象棋 AlphaGo 风格完整对局【自我对弈】CoT 强化学习 (Zero-Engine Self-Play RL)"
echo "============================================================================"
python3 train_rl_selfplay.py "$@"
