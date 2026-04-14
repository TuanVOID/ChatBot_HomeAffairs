@echo off
chcp 65001 >nul
cd /d "f:\SpeechToText-indti\ChatBot2_Opus\scripts\CrawlData"

echo ============================================================
echo   START CAMOUFOX 3 WORKERS (PARALLEL MODE)
echo ============================================================
echo.
echo   Mode:
echo     - Chay 3 worker cung luc (w1, w2, w3)
echo     - Tu tiep tuc tu tien do dang do (uu tien state Camoufox/Scrapling)
echo.
echo   Delay:
echo     - Dang dat mac dinh 15s (random +-2s trong worker)
echo       de giam xac suat dinh Cloudflare
echo.
echo   Log:
echo     logs\camoufox_parallel\w1.log
echo     logs\camoufox_parallel\w2.log
echo     logs\camoufox_parallel\w3.log
echo ============================================================
echo.

if not exist logs\camoufox_parallel mkdir logs\camoufox_parallel

set BASE_DELAY=15
set PROXY_W1=
set PROXY_W2=
set PROXY_W3=

python 08_parallel_camoufox.py ^
  --delay %BASE_DELAY% ^
  --cf-manual-wait 30 ^
  --captcha-manual-wait 30 ^
  --captcha-retries 8 ^
  --navigation-retries 4 ^
  --proxy-w1 "%PROXY_W1%" ^
  --proxy-w2 "%PROXY_W2%" ^
  --proxy-w3 "%PROXY_W3%"

echo.
echo ============================================================
echo   Camoufox parallel da dung.
echo   Chay lai file nay de tiep tuc.
echo ============================================================
pause
