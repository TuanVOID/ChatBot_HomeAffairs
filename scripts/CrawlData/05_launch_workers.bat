@echo off
chcp 65001 >nul
echo ============================================================
echo   KHOI TAO 3 CHROME WORKERS (Port 9222 - 9224)
echo ============================================================
echo.
echo   QUAN TRONG: Tat het Chrome truoc khi chay file nay!
echo.
echo   Profile:
echo     - W1: C:\ChromeV1\w1
echo     - W2: C:\ChromeV1\w2
echo     - W3: C:\ChromeV1\w3
echo ============================================================

echo.
echo [1/3] Mo Worker 1 (Port 9222)...
start "" "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222 --user-data-dir=C:\ChromeV1\w1 --no-first-run --no-default-browser-check
timeout /t 5 /nobreak >nul

echo [2/3] Mo Worker 2 (Port 9223)...
start "" "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9223 --user-data-dir=C:\ChromeV1\w2 --no-first-run --no-default-browser-check
timeout /t 5 /nobreak >nul

echo [3/3] Mo Worker 3 (Port 9224)...
start "" "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9224 --user-data-dir=C:\ChromeV1\w3 --no-first-run --no-default-browser-check
timeout /t 3 /nobreak >nul

echo.
echo ============================================================
echo   XONG! Kiem tra xem co 3 cua so Chrome khong.
echo.
echo   Buoc tiep:
echo     1. Vao thuvienphapluat.vn tren moi cua so
echo     2. Pass Cloudflare cho tung cai
echo     3. Chay 05_start_crawl.bat
echo ============================================================
pause
