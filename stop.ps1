# Legal RAG Chatbot - Stop All Services
# Usage:
#   .\stop.ps1               # Tat server + ngrok (giu Ollama)
#   .\stop.ps1 -All          # Tat tat ca ke ca Ollama
#   .\stop.ps1 -OllamaOnly   # Chi tat Ollama

param(
    [switch]$All,
    [switch]$OllamaOnly
)

$ErrorActionPreference = "Continue"

function Write-Step($msg) { Write-Host "[*] $msg" -ForegroundColor Yellow }
function Write-Ok($msg) { Write-Host "[OK] $msg" -ForegroundColor Green }
function Write-Err($msg) { Write-Host "[ERR] $msg" -ForegroundColor Red }
function Write-Info($msg) { Write-Host "[i] $msg" -ForegroundColor Gray }

Write-Host "`n$('=' * 60)" -ForegroundColor Red
Write-Host "  LEGAL RAG CHATBOT - SHUTDOWN" -ForegroundColor Red
Write-Host "$('=' * 60)`n" -ForegroundColor Red

$killed = 0

# ---- STEP 1: Tat FastAPI server ----
if (-not $OllamaOnly) {
    Write-Step "Tim va tat FastAPI server..."

    $serverProcs = Get-Process python -ErrorAction SilentlyContinue |
    Where-Object {
        try {
            $cmdline = (Get-CimInstance Win32_Process -Filter "ProcessId=$($_.Id)" -ErrorAction SilentlyContinue).CommandLine
            $cmdline -match "server\.py"
        }
        catch { $false }
    }

    if ($serverProcs) {
        foreach ($p in $serverProcs) {
            try {
                Stop-Process -Id $p.Id -Force
                Write-Ok "Da tat server (PID: $($p.Id))"
                $killed++
            }
            catch {
                Write-Err "Khong the tat PID $($p.Id): $_"
            }
        }
    }
    else {
        Write-Info "Khong tim thay server dang chay"
    }
}

# ---- STEP 2: Dong ngrok tunnels ----
if (-not $OllamaOnly) {
    Write-Step "Dong ngrok tunnels..."

    $tunnelsClosed = 0
    try {
        $response = Invoke-RestMethod -Uri "http://127.0.0.1:4040/api/tunnels" -TimeoutSec 2 -ErrorAction Stop
        foreach ($tunnel in $response.tunnels) {
            $uri = $tunnel.uri
            if ($uri) {
                try {
                    $null = Invoke-RestMethod -Uri "http://127.0.0.1:4040$uri" -Method Delete -TimeoutSec 2 -ErrorAction Stop
                    $tunnelsClosed++
                }
                catch { }
            }
        }
        if ($tunnelsClosed -gt 0) {
            Write-Ok "Da dong $tunnelsClosed ngrok tunnel(s)"
        }
    }
    catch {
        Write-Info "Khong co ngrok API dang chay"
    }

    # Kill ngrok process
    $ngrokProcs = Get-Process ngrok -ErrorAction SilentlyContinue
    if ($ngrokProcs) {
        foreach ($p in $ngrokProcs) {
            try {
                Stop-Process -Id $p.Id -Force
                Write-Ok "Da tat ngrok (PID: $($p.Id))"
                $killed++
            }
            catch { }
        }
    }
    elseif ($tunnelsClosed -eq 0) {
        Write-Info "Khong co ngrok dang chay"
    }
}

# ---- STEP 3: Tat Ollama (neu -All hoac -OllamaOnly) ----
if ($All -or $OllamaOnly) {
    Write-Step "Tat Ollama..."

    # Tat ollama runners (model processes)
    $runners = Get-Process ollama_llama_server -ErrorAction SilentlyContinue
    if ($runners) {
        foreach ($p in $runners) {
            try {
                Stop-Process -Id $p.Id -Force
                $killed++
            }
            catch { }
        }
        Write-Ok "Da tat $($runners.Count) ollama runner(s)"
    }

    # Tat ollama app/serve
    $ollamaProcs = Get-Process ollama*, Ollama* -ErrorAction SilentlyContinue |
    Where-Object { $_.ProcessName -notmatch "ollama_llama_server" }
    if ($ollamaProcs) {
        foreach ($p in $ollamaProcs) {
            try {
                Stop-Process -Id $p.Id -Force
                Write-Ok "Da tat $($p.ProcessName) (PID: $($p.Id))"
                $killed++
            }
            catch { }
        }
    }
    else {
        Write-Info "Khong tim thay Ollama dang chay"
    }
}
else {
    Write-Info "Giu Ollama chay (dung -All de tat luon Ollama)"
}

# ---- Cleanup ----
$pidFile = Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) ".server.pid"
if (Test-Path $pidFile) {
    Remove-Item $pidFile -Force
}

Write-Host "`n$('=' * 60)" -ForegroundColor Green
if ($killed -gt 0) {
    Write-Host "  Da dung $killed process(es)" -ForegroundColor Green
}
else {
    Write-Host "  Khong co gi can dung" -ForegroundColor Yellow
}
Write-Host "$('=' * 60)`n" -ForegroundColor Green
