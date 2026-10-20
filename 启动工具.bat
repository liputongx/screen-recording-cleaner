@echo off
chcp 65001 >nul
title 录屏文档清洗工具
echo ============================================
echo   录屏文档清洗工具 启动中...
echo   浏览器将自动打开 http://127.0.0.1:8765
echo ============================================

rem 优先使用本机 venv 中的 Python（若存在），否则回退到系统 python
set "VENV_PY=%USERPROFILE%\.workbuddy\binaries\python\envs\default\Scripts\python.exe"
if exist "%VENV_PY%" (
    "%VENV_PY%" app.py
) else (
    python app.py
)
pause
