@echo off
chcp 65001 >nul
cd /d "f:\SpeechToText-indti\ChatBot2_Opus\scripts\CrawlData"

echo ============================================================
echo   INSTALL CAMOUFOX CRAWL DEPENDENCIES
echo ============================================================
echo.
echo   1. Install Camoufox (stable)
echo   2. Fetch Camoufox browser binary
echo.
echo   Note:
echo     - OCR still uses local Tesseract at:
echo       C:\Program Files\Tesseract-OCR\tesseract.exe
echo ============================================================
echo.

python -m pip install -U "camoufox[geoip]"
if errorlevel 1 (
  echo [ERROR] pip install camoufox failed.
  pause
  exit /b 1
)

python -m camoufox fetch
if errorlevel 1 (
  echo [ERROR] camoufox fetch failed.
  pause
  exit /b 1
)

echo.
echo ============================================================
echo   CAMOUFOX INSTALL COMPLETED
echo ============================================================
pause
