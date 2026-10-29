@echo off
chcp 65001 >nul
echo ============================================================================
echo   中国象棋 后训练强化学习探索 (DeepSeek-R1 / DAPO GRPO CoT Post-Training)
echo   默认基于 checkpoints/best_model.pt 进行 2,000 Steps 强化学习探索 (不走时间额度)
echo   支持续训追加 Step，例如: run_rl_windows.bat --resume --add-steps 3000
echo ============================================================================

python train_rl_grpo.py %*
pause
