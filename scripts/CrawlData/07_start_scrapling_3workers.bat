@echo off
chcp 65001 >nul
cd /d "f:\SpeechToText-indti\ChatBot2_Opus\scripts\CrawlData"

echo ============================================================
echo   START SCRAPLING ROUND-ROBIN (SEQUENTIAL 1 WORKER/TIME)
echo ============================================================
echo.
echo   Muc tieu:
echo     - Giam xac suat dinh Cloudflare so voi mode parallel.
echo     - Luan phien W1 -> W2 -> W3, moi worker 10 trang/luot.
echo.
echo   Yeu cau:
echo     1. Da mo 3 Chrome workers (9222-9224)
echo     2. Da pass Cloudflare tren profile
echo.
echo   Tien do RR luu tai:
echo     state\scrapling_rr3_state.json
echo.
echo   Log RR:
echo     logs\scrapling_rr\w1.log, w2.log, w3.log
echo.
echo   Neu muon chay song song 3 worker:
echo     07_start_scrapling_3workers_parallel.bat
echo ============================================================

if not exist logs\scrapling_rr mkdir logs\scrapling_rr

set START_FROM_WORKER=1
set PAGES_PER_TURN=10

python 07_round_robin_scrapling.py ^
  --pages-per-turn %PAGES_PER_TURN% ^
  --delay 18 ^
  --cf-manual-wait 30 ^
  --max-blocked-retries 8 ^
  --delay-between-workers 2 ^
  --start-from-worker %START_FROM_WORKER% ^
  --dynamic

echo.
echo ============================================================
echo   Da dung round-robin Scrapling.
echo   Chay lai file nay de resume tiep.
echo ============================================================
pause

