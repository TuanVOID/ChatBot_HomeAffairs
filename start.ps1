# Legal RAG Chatbot - Start All Services
# Usage:
#   .\start.ps1              # Local only
#   .\start.ps1 -Ngrok       # With ngrok tunnel
#   .\start.ps1 -Port 9000   # Custom port

param(
    [switch]$Ngrok,
    [int]$Port = 8899,
    [string]$Host_ = "0.0.0.0"
)

$ErrorActionPreference = "Continue"
$PROJECT_ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path

function Write-Header($msg) { Write-Host "`n$('=' * 60)" -ForegroundColor Cyan; Write-Host "  $msg" -ForegroundColor Cyan; Write-Host "$('=' * 60)" -ForegroundColor Cyan }
function Write-Step($msg) { Write-Host "[*] $msg" -ForegroundColor Yellow }
function Write-Ok($msg) { Write-Host "[OK] $msg" -ForegroundColor Green }
function Write-Err($msg) { Write-Host "[ERR] $msg" -ForegroundColor Red }
function Write-Info($msg) { Write-Host "[i] $msg" -ForegroundColor Gray }

# Load .env
$envFile = Join-Path $PROJECT_ROOT ".env"
if (Test-Path $envFile) {
    Get-Content $envFile -Encoding UTF8 | ForEach-Object {
        if ($_ -match '^\s*([^#][^=]+?)\s*=\s*(.+)\s*$') {
            [Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process")
        }
    }
}

$OLLAMA_URL = if ($env:OLLAMA_BASE_URL) { $env:OLLAMA_BASE_URL } else { "http://localhost:11434" }
$CHAT_MODEL = if ($env:CHAT_MODEL) { $env:CHAT_MODEL } else { "qwen2.5:7b-instruct" }

Write-Header "LEGAL RAG CHATBOT - STARTUP"
Write-Info "Project: $PROJECT_ROOT"
Write-Info "Ollama: $OLLAMA_URL | Model: $CHAT_MODEL | Port: $Port"

# ---- STEP 1: Ollama ----
Write-Step "Kiem tra Ollama..."

$ollamaRunning = $false
try {
    $null = Invoke-RestMethod -Uri "$OLLAMA_URL/api/tags" -TimeoutSec 3 -ErrorAction Stop
    $ollamaRunning = $true
    Write-Ok "Ollama dang chay"
}
catch {
    Write-Info "Ollama chua chay, dang khoi dong..."

    $ollamaCmd = Get-Command ollama -ErrorAction SilentlyContinue
    if (-not $ollamaCmd) {
        $defaultPath = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
        if (Test-Path $defaultPath) {
            $ollamaCmd = $defaultPath
        }
        else {
            Write-Err "Khong tim thay ollama! Cai tu: https://ollama.com/download"
            exit 1
        }
    }

    Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
    Write-Info "Da start ollama serve (background)"

    $maxWait = 30
    $waited = 0
    while ($waited -lt $maxWait) {
        Start-Sleep -Seconds 1
        $waited++
        try {
            $null = Invoke-RestMethod -Uri "$OLLAMA_URL/api/tags" -TimeoutSec 2 -ErrorAction Stop
            $ollamaRunning = $true
            break
        }
        catch { }
        if ($waited % 5 -eq 0) { Write-Info "Dang doi Ollama... ($waited/${maxWait}s)" }
    }

    if ($ollamaRunning) {
        Write-Ok "Ollama da san sang! (sau ${waited}s)"
    }
    else {
        Write-Err "Ollama khong khoi dong duoc sau ${maxWait}s"
        exit 1
    }
}

# ---- STEP 2: Check/Pull model ----
Write-Step "Kiem tra model $CHAT_MODEL..."

$modelExists = $false
try {
    $tags = Invoke-RestMethod -Uri "$OLLAMA_URL/api/tags" -TimeoutSec 5 -ErrorAction Stop
    foreach ($m in $tags.models) {
        if ($m.name -eq $CHAT_MODEL -or $m.name -eq "${CHAT_MODEL}:latest") {
            $modelExists = $true
            break
        }
    }
}
catch {
    Write-Err "Khong the kiem tra model list"
}

if ($modelExists) {
    Write-Ok "Model $CHAT_MODEL da co san"
}
else {
    Write-Info "Model $CHAT_MODEL chua co, dang pull... (lan dau ~4-5GB)"
    try {
        & ollama pull $CHAT_MODEL
        if ($LASTEXITCODE -eq 0) {
            Write-Ok "Pull model thanh cong!"
        }
        else {
            Write-Err "Pull model that bai"
            Write-Info "Ban co the pull thu cong: ollama pull $CHAT_MODEL"
        }
    }
    catch {
        Write-Err "Loi pull model: $_"
    }
}

# ---- STEP 3: Check port ----
Write-Step "Kiem tra port $Port..."
# Filter out TimeWait/CloseWait (PID 0) - chi block khi co process thuc su dang listen
$portInUse = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue |
Where-Object { $_.OwningProcess -gt 0 -and $_.State -ne 'TimeWait' -and $_.State -ne 'CloseWait' }
if ($portInUse) {
    $pid_ = ($portInUse | Select-Object -First 1).OwningProcess
    $proc = Get-Process -Id $pid_ -ErrorAction SilentlyContinue
    Write-Err "Port $Port dang bi chiem boi: $($proc.ProcessName) (PID: $pid_)"
    Write-Info "Chay .\stop.ps1 truoc, hoac dung -Port <khac>"
    exit 1
}
Write-Ok "Port $Port san sang"

# ---- STEP 4: Start server ----
Write-Header "KHOI DONG SERVER"

$serverArgs = @("server.py", "--port", $Port, "--host", $Host_)
if ($Ngrok) {
    $serverArgs += "--ngrok"
    Write-Info "Mode: Server + Ngrok tunnel"
}
else {
    Write-Info "Mode: Local only (them -Ngrok de expose qua internet)"
}

Write-Info "Nhan Ctrl+C de dung server"

try {
    $env:PYTHONUNBUFFERED = "1"
    Set-Location $PROJECT_ROOT
    & python @serverArgs
}
finally {
    Write-Info "Server da dung."
}
