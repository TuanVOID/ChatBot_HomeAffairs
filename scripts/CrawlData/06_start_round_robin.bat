@echo off
chcp 65001 >nul
cd /d "f:\SpeechToText-indti\ChatBot2_Opus\scripts\CrawlData"

echo ============================================================
echo   ROUND-ROBIN CRAWL 3 WORKERS (W1 -> W2 -> W3)
echo ============================================================
echo.
echo   Workload:
echo     - W1 + W2: Co quan TW (chia 4 range)
echo     - W3: 12 co quan con lai
echo.
echo   State moi:
echo     state\rr3_workers_state.json
echo.
echo   Script se tu migrate state cu cua W1/W2 (neu co):
echo     state\rr_workers_state.json -> state\rr3_workers_state.json
echo.
echo   Dung bang Ctrl+C, chay lai file nay de resume.
echo ============================================================
echo.

set START_FROM_WORKER=1
python 06_round_robin_crawl.py --pages-per-turn 10 --delay 10 --delay-between-workers 2 --start-from-worker %START_FROM_WORKER%
REM Vi du: set START_FROM_WORKER=2 de chay theo thu tu 2-3-1

echo.
echo ============================================================
echo   Da dung crawler round-robin 3 workers.
echo   Neu can tiep tuc, chay lai file nay (tu dong resume).
echo ============================================================
pause
