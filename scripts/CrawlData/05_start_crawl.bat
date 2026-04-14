@echo off
chcp 65001 >nul
cd /d "f:\SpeechToText-indti\ChatBot2_Opus\scripts\CrawlData"

echo ============================================================
echo   START CRAWL - 3 WORKERS MODE
echo ============================================================
echo.
echo   Luong moi:
echo     1. Chay 05_launch_workers.bat de mo 3 Chrome profile
echo     2. Chay file nay de bat dau round-robin crawl
echo.
echo   File nay se goi:
echo     06_start_round_robin.bat
echo ============================================================
echo.

call 06_start_round_robin.bat
