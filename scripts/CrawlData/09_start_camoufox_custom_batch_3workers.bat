@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul
cd /d "f:\SpeechToText-indti\ChatBot2_Opus\scripts\CrawlData"

echo ============================================================
echo   CAMOUFOX CUSTOM BATCH (3 WORKERS PARALLEL + RESUME)
echo ============================================================
echo.
echo   Dinh dang moi worker:
echo     so_bai,link
echo   Vi du:
echo     120,https://...org=12...page=1
echo.
echo   Script tu tinh so page can crawl theo cong thuc:
echo     pages = ceil(so_bai / 20)
echo   va bat dau tu page trong link ban dua.
echo ============================================================
echo.

if not exist logs\camoufox_custom_batch mkdir logs\camoufox_custom_batch

set BASE_DELAY=11
set VIEWPORT_WIDTH=1600
set VIEWPORT_HEIGHT=900
set OUTPUT_DIR=output
set RESUME_STATE_DIR=state\custom_batch_resume
set RESET_RESUME=0
set RESET_FLAG=
if "%RESET_RESUME%"=="1" set RESET_FLAG=--reset-resume

rem ============================================================
rem W1 TASKS (so_bai,link)
rem ============================================================
set "W1_TASK_1=360,https://thuvienphapluat.vn/page/tim-van-ban.aspx?keyword=&area=0&match=True&type=0&status=0&signer=0&edate=13/04/2026&sort=1&lan=1&scan=0&org=22&fields=&page=1"
set "W1_TASK_2="
set "W1_TASK_3="

rem ============================================================
rem W2 TASKS (so_bai,link)
rem ============================================================
set "W2_TASK_1=200,https://thuvienphapluat.vn/page/tim-van-ban.aspx?keyword=&area=0&match=True&type=0&status=0&signer=0&sort=1&lan=1&scan=0&org=26&fields=&page=1"
set "W2_TASK_2="
set "W2_TASK_3="

rem ============================================================
rem W3 TASKS (so_bai,link)
rem ============================================================
set "W3_TASK_1=400,https://thuvienphapluat.vn/page/tim-van-ban.aspx?keyword=&area=0&match=True&type=0&status=0&signer=0&edate=13/04/2026&sort=1&lan=1&scan=0&org=33&fields=&page=1"
set "W3_TASK_2="
set "W3_TASK_3="

rem Proxy per worker (sua tai day neu can):
set PROXY_W1=118.70.187.215:31961:VuMxKk:HWxJyo
set PROXY_W2=118.70.171.104:14792:VuMxKk:HWxJyo
set PROXY_W3=118.70.171.121:47703:VuMxKk:HWxJyo

call :BUILD_PLAN W1_PLAN W1_TASK_1 W1_TASK_2 W1_TASK_3
call :BUILD_PLAN W2_PLAN W2_TASK_1 W2_TASK_2 W2_TASK_3
call :BUILD_PLAN W3_PLAN W3_TASK_1 W3_TASK_2 W3_TASK_3

echo Output dir: %OUTPUT_DIR%
echo Resume state dir: %RESUME_STATE_DIR%
echo Reset resume: %RESET_RESUME%
echo.
echo W1_PLAN=%W1_PLAN%
echo W2_PLAN=%W2_PLAN%
echo W3_PLAN=%W3_PLAN%
echo.

python 09_parallel_camoufox_custom_batch.py ^
  --delay %BASE_DELAY% ^
  --viewport-width %VIEWPORT_WIDTH% ^
  --viewport-height %VIEWPORT_HEIGHT% ^
  --output-dir "%OUTPUT_DIR%" ^
  --resume-state-dir "%RESUME_STATE_DIR%" ^
  --cf-manual-wait 30 ^
  --captcha-manual-wait 30 ^
  --captcha-retries 8 ^
  --navigation-retries 4 ^
  --fresh-profiles ^
  --plan-w1 "%W1_PLAN%" ^
  --plan-w2 "%W2_PLAN%" ^
  --plan-w3 "%W3_PLAN%" ^
  --proxy-w1 "%PROXY_W1%" ^
  --proxy-w2 "%PROXY_W2%" ^
  --proxy-w3 "%PROXY_W3%" ^
  %RESET_FLAG%

echo.
echo ============================================================
echo   Batch da dung.
echo   Resume state:
echo     state\custom_batch_resume\w1.json
echo     state\custom_batch_resume\w2.json
echo     state\custom_batch_resume\w3.json
echo   Logs:
echo     logs\camoufox_custom_batch\w1.log
echo     logs\camoufox_custom_batch\w2.log
echo     logs\camoufox_custom_batch\w3.log
echo ============================================================
pause
goto :eof

:BUILD_PLAN
set "OUTVAR=%~1"
shift
set "PLAN="
:BUILD_PLAN_LOOP
if "%~1"=="" goto BUILD_PLAN_DONE
call set "ITEM=%%%~1%%"
if defined ITEM (
  if defined PLAN (
    set "PLAN=!PLAN!;!ITEM!"
  ) else (
    set "PLAN=!ITEM!"
  )
)
shift
goto BUILD_PLAN_LOOP
:BUILD_PLAN_DONE
set "%OUTVAR%=%PLAN%"
goto :eof

