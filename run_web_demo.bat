@echo off
echo ========================================
echo Mochi 情感分析系统 - Web Demo 启动器
echo ========================================
echo.

echo [1/3] 检查依赖...
pip show streamlit >nul 2>&1
if errorlevel 1 (
    echo 未检测到 streamlit，正在安装依赖...
    pip install -r requirements_web.txt
) else (
    echo 依赖已安装 ✓
)

echo.
echo [2/3] 检查模型文件...
if exist "checkpoints\3class_final\best_model.pth" (
    echo 模型文件已就绪 ✓
) else (
    echo 警告: 模型文件未找到，将使用随机初始化权重（仅供演示）
)

echo.
echo [3/3] 启动 Web Demo...
echo 访问地址: http://localhost:8501
echo 按 Ctrl+C 停止服务
echo.
streamlit run web_demo_upgraded.py

pause
