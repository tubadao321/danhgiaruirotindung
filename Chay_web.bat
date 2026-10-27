
@echo off
title CreditRisk AI - Nhom 14
cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo Khong tim thay moi truong Python .venv!
    echo Hay kiem tra lai thu muc project.
    pause
    exit /b 1
)

call ".venv\Scripts\activate.bat"
python -m streamlit run app.py

pause
