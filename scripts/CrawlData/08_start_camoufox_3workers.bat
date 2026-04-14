@echo off
chcp 65001 >nul
cd /d "f:\SpeechToText-indti\ChatBot2_Opus\scripts\CrawlData"

echo ============================================================
echo   START CAMOUFOX ROUND-ROBIN (3 WORKERS, SEQUENTIAL)
echo ============================================================
echo.
echo   Khong can mo Chrome debug profiles 9222-9224 nua.
echo   Camoufox se tu tao profile rieng:
echo     state\camoufox_profiles\w1
echo     state\camoufox_profiles\w2
echo     state\camoufox_profiles\w3
echo.
echo   State round-robin:
echo     state\camoufox_rr3_state.json
echo.
echo   Log worker:
echo     logs\camoufox_rr\w1.log, w2.log, w3.log
echo.
echo   Neu muon chay 3 workers song song:
echo     08_start_camoufox_3workers_parallel.bat
echo ============================================================
echo.

if not exist logs\camoufox_rr mkdir logs\camoufox_rr

set START_FROM_WORKER=1
set PAGES_PER_TURN=10

python 08_round_robin_camoufox.py ^
  --pages-per-turn %PAGES_PER_TURN% ^
  --delay 18 ^
  --cf-manual-wait 30 ^
  --captcha-manual-wait 30 ^
  --captcha-retries 8 ^
  --navigation-retries 4 ^
  --delay-between-workers 2 ^
  --start-from-worker %START_FROM_WORKER%

echo.
echo ============================================================
echo   Da dung round-robin Camoufox.
echo   Chay lai file nay de resume tiep.
echo ============================================================
pause
