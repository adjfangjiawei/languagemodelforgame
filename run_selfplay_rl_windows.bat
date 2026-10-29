@echo off
chcp 65001 >nul
echo ============================================================================
echo   中国象棋 AlphaGo 风格完整对局【自我对弈】CoT 强化学习 (Zero-Engine Self-Play RL)
echo   模型在完整红黑对局中自己与自己/历史冠军对弈，每步先推演 CoT 再落子，按真实胜负演化！
echo   支持强制指定 Step 与断点追加，例如: run_selfplay_rl_windows.bat --resume --add-steps 2000
echo ============================================================================

python train_rl_selfplay.py %*
pause
