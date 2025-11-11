@echo off
REM =============================================================================
REM Environment Setup Script for Image Similarity API (Windows)
REM =============================================================================

echo ==========================================
echo Image Similarity API - Environment Setup
echo ==========================================
echo.

REM Check if argument is provided
if "%1"=="" (
    echo Error: No environment specified
    echo.
    goto :show_usage
)

set ENV_TYPE=%1

REM Determine source file
if /i "%ENV_TYPE%"=="dev" set SOURCE_FILE=.env.development
if /i "%ENV_TYPE%"=="development" set SOURCE_FILE=.env.development
if /i "%ENV_TYPE%"=="prod" set SOURCE_FILE=.env.production
if /i "%ENV_TYPE%"=="production" set SOURCE_FILE=.env.production
if /i "%ENV_TYPE%"=="ubuntu" set SOURCE_FILE=.env.ubuntu
if /i "%ENV_TYPE%"=="windows" set SOURCE_FILE=.env.windows

if not defined SOURCE_FILE (
    echo Error: Invalid environment type: %ENV_TYPE%
    echo.
    goto :show_usage
)

REM Check if source file exists
if not exist "%SOURCE_FILE%" (
    echo Error: Configuration file %SOURCE_FILE% not found
    echo.
    echo Please make sure you have the following files:
    echo   - .env.development
    echo   - .env.production
    echo   - .env.ubuntu
    echo   - .env.windows
    echo.
    echo You can copy from .env.example and modify as needed.
    exit /b 1
)

REM Backup existing .env if it exists
if exist ".env" (
    echo Backing up existing .env to .env.backup...
    copy /Y .env .env.backup >nul
)

REM Copy the selected environment file
echo Setting up %ENV_TYPE% environment...
copy /Y "%SOURCE_FILE%" .env >nul

REM Create necessary directories
echo Creating necessary directories...
if not exist logs mkdir logs
if not exist models mkdir models
if not exist index mkdir index
if not exist shared\uploads mkdir shared\uploads

echo.
echo ==========================================
echo Setup Complete!
echo ==========================================
echo Environment: %ENV_TYPE%
echo Configuration file: %SOURCE_FILE% -^> .env
echo.
echo Next steps:
echo 1. Verify your .env file has correct values
echo 2. Install dependencies: pip install -r requirements.txt
echo 3. Setup the model: python setup_model.py
echo 4. Run the API: uvicorn main:app --reload
echo.
echo To test the setup, run: python main.py
echo.
exit /b 0

:show_usage
echo Usage: setup_env.bat [environment]
echo.
echo Available environments:
echo   dev         - Development environment
echo   prod        - Production environment
echo   ubuntu      - Ubuntu-specific configuration
echo   windows     - Windows-specific configuration
echo.
echo Example:
echo   setup_env.bat dev
echo   setup_env.bat windows
echo.
exit /b 1
