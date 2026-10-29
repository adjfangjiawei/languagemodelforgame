@echo off
chcp 65001 >nul
echo ============================================================================
echo   中国象棋 CoT 增量生成新数据 + 强制指定 Step 续训 (不走时间额度!)
echo   默认追加生成 5,000 个新局面 (~10,000条镜像CoT)，并强制续训 +12,000 Steps
echo   也可传参覆盖，例如: continue_sft_windows.bat --add-positions 10000 --add-steps 25000
echo ============================================================================

python continue_sft.py %*
pause
