@echo off
REM Setup script for bite-o-serve
REM This script installs the application and its dependencies using uv

REM Check if uv is installed
where uv >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Installing package mannager...
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
)
call uv self update

echo Creating virtual environment...
call uv venv

REM Activate virtual environment
echo Activating virtual environment...
call .\.venv\Scripts\activate

REM Install dependencies
echo Installing dependencies...
call uv pip install -e .

echo =================-Installation complete!-=================
echo Running SPOROTRACKER
call uv run main.py

REM 
pause