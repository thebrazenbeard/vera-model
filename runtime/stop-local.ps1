# Stop ONLY the model-server process owned by this dedicated local runtime.
$ErrorActionPreference = 'Stop'
$root = Join-Path $env:LOCALAPPDATA 'VeraModel\glm24b'
$pidFile = Join-Path $root 'llama.pid'
if (-not (Test-Path $pidFile)) { Write-Output 'MODEL_SERVER_NOT_REGISTERED'; exit 0 }
$targetPid = [int](Get-Content $pidFile)
$process = Get-CimInstance Win32_Process -Filter "ProcessId=$targetPid" -ErrorAction SilentlyContinue
if (-not $process) { Write-Output 'MODEL_SERVER_ALREADY_STOPPED'; exit 0 }
$expected = 'Dolphin-Mistral-GLM-4.7-Flash-24B-Venice-Edition-Thinking-Uncensored.i1-IQ2_XS.gguf'
if ($process.Name -ne 'llama-server.exe' -or
    $process.CommandLine -notlike "*$expected*" -or
    $process.CommandLine -notlike '*--port 8097*') {
    throw 'PID does not refer to the exact dedicated model server; no termination'
}
Stop-Process -Id $targetPid -ErrorAction Stop
Write-Output ('STOPPED_DEDICATED_MODEL_SERVER_PID=' + $targetPid)
