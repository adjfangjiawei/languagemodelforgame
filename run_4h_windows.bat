@echo off
chcp 65001 >nul
echo ======================================================================
echo   Xiangqi Variation-DAG CoT LM - GTX 1660 Ti (6GB) ~4h One-Click Run
echo ======================================================================

python -c "import torch; assert torch.cuda.is_available(), 'CUDA not active'" 2>nul
if %errorlevel% neq 0 (
    echo [提示] 当前 Python 环境未检测到可用 CUDA PyTorch，正在尝试安装 CUDA 12.1 版 PyTorch...
    pip install torch --index-url https://download.pytorch.org/whl/cu121
)

pip install -r requirements.txt

python run_4h_pipeline.py --total-hours 4.0 --data-minutes 35 --preset fast-4m
echo.
echo [完成] 训练与评测已结束！模型权重保存在 checkpoints\best_model.pt
pause
