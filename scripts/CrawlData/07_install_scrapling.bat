@echo off
chcp 65001 >nul
cd /d "f:\SpeechToText-indti\ChatBot2_Opus"

echo ============================================================
echo   INSTALL SCRAPLING (LOCAL REPO)
echo ============================================================
echo.
echo   Step 1: Install Scrapling editable + fetchers extras
python -m pip install -e "03.Scrapling[fetchers]"
if errorlevel 1 goto :fail

echo.
echo   Step 2: Install browser runtime for Scrapling/Playwright
scrapling install
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo   DONE.
echo ============================================================
pause
exit /b 0

:fail
echo.
echo ============================================================
echo   INSTALL FAILED.
echo ============================================================
pause
exit /b 1
