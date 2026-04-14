@echo off
chcp 65001 >nul
cd /d "f:\SpeechToText-indti\ChatBot2_Opus\scripts\CrawlData"

echo ============================================================
echo   START SCRAPLING CRAWL - 3 WORKERS (PARALLEL)
echo ============================================================
echo.
echo   Yeu cau:
echo   1. Da mo 3 Chrome workers (port 9222-9224)
echo   2. Da pass Cloudflare tren tung profile
echo.
echo   Co the mo Chrome bang:
echo     05_launch_workers.bat
echo   Luu y:
echo     - Worker uu tien dung context profile CDP hien co (giu cookie/session).
echo     - Neu muon dung proxy, hay setup proxy ngay trong profile Chrome (extension/system proxy),
echo       hoac them --force-proxy-context vao tung worker (de giam on dinh).
echo     - Theo doi log tai: logs\scrapling\w1.log, w2.log, w3.log
echo ============================================================

if not exist logs\scrapling mkdir logs\scrapling

echo [1/3] Worker 1 - Co quan TW pages 1-50,101-150
start "W1-Scrapling" cmd /k python 07_scrapling_crawl_by_org.py --worker w1 --port 9222 --orgs 1 --ranges 1-50,101-150 --delay 18 --dynamic --cf-manual-wait 30 --max-blocked-retries 8 ^>^> logs\scrapling\w1.log 2^>^&1
timeout /t 3 /nobreak >nul

echo [2/3] Worker 2 - Co quan TW pages 51-100,151-200
start "W2-Scrapling" cmd /k python 07_scrapling_crawl_by_org.py --worker w2 --port 9223 --orgs 1 --ranges 51-100,151-200 --delay 18 --dynamic --cf-manual-wait 30 --max-blocked-retries 8 ^>^> logs\scrapling\w2.log 2^>^&1
timeout /t 3 /nobreak >nul

echo [3/3] Worker 3 - 12 co quan con lai
start "W3-Scrapling" cmd /k python 07_scrapling_crawl_by_org.py --worker w3 --port 9224 --orgs 3,12,6,19,22,23,26,33,95,97,98,104 --ranges 1-999 --delay 18 --dynamic --cf-manual-wait 30 --max-blocked-retries 8 ^>^> logs\scrapling\w3.log 2^>^&1

echo.
echo ============================================================
echo   Da start 3 workers Scrapling (parallel).
echo.
echo   Resume:
echo     Chay lai dung lenh tren, spider tu resume theo crawldir.
echo.
echo   Output:
echo     output\org_*.jsonl
echo ============================================================
pause

