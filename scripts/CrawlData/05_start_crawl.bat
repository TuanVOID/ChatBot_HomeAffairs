@echo off
chcp 65001 >nul
cd /d "f:\SpeechToText-indti\ChatBot2_Opus\scripts\CrawlData"

echo ============================================================
echo   START CRAWL - CAMOUFOX 3 WORKERS PARALLEL MODE
echo ============================================================
echo.
echo   Luong mac dinh moi:
echo     1. (Lan dau) chay 08_install_camoufox.bat
echo     2. Chay file nay de bat dau crawl song song 3 workers
echo.
echo   File nay se goi:
echo     08_start_camoufox_3workers_parallel.bat
echo.
echo   Neu can mode luan phien (sequential):
echo     08_start_camoufox_3workers.bat
echo ============================================================
echo.

call 08_start_camoufox_3workers_parallel.bat
