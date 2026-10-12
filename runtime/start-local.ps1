# Launch a verified GLM24B IQ2_XS GGUF in a loopback-only llama.cpp sidecar.
# No VeraOS, Ollama, P.O.R.T.A.L., or system-startup mutation.
$ErrorActionPreference = 'Stop'
$source = 'D:\VERA\models\gguf\vera-glm24b-iq2-xs'
$name = 'Dolphin-Mistral-GLM-4.7-Flash-24B-Venice-Edition-Thinking-Uncensored.i1-IQ2_XS.gguf'
$model = Join-Path $source $name
$expectedSize = [int64]7207034784
$expectedSha = '563d47ae5fb99f5408043440eb4b545aee0239a50c2d0439e83abefa7ce8edba'
if (-not (Test-Path -LiteralPath $model -PathType Leaf)) { throw 'Verified GGUF not present' }
if ((Get-Item -LiteralPath $model).Length -ne $expectedSize) { throw 'GGUF size mismatch' }
$observed = (Get-FileHash -LiteralPath $model -Algorithm SHA256).Hash.ToLowerInvariant()
if ($observed -ne $expectedSha) { throw 'GGUF SHA-256 mismatch' }
$existing = Get-NetTCPConnection -State Listen -LocalPort 8097 -ErrorAction SilentlyContinue
if ($existing) { throw 'Dedicated model port 8097 already occupied' }
$server = Get-Command llama-server.exe -ErrorAction Stop
$localRoot = Join-Path $env:LOCALAPPDATA 'VeraModel\glm24b'
if (-not (Test-Path $localRoot)) { New-Item -ItemType Directory -Path $localRoot | Out-Null }
$keyFile = Join-Path $localRoot 'api-key.txt'
if (-not (Test-Path -LiteralPath $keyFile)) {
    $bytes = New-Object byte[] 32
    $rng = [System.Security.Cryptography.RandomNumberGenerator]::Create()
    try { $rng.GetBytes($bytes) } finally { $rng.Dispose() }
    $key = -join ($bytes | ForEach-Object { $_.ToString('x2') })
    [System.IO.File]::WriteAllText($keyFile, $key + [Environment]::NewLine)
    & icacls.exe $keyFile /inheritance:r /grant:r "$($env:USERNAME):(R,W)" | Out-Null
}
$logOut = Join-Path $localRoot 'llama.stdout.log'
$logErr = Join-Path $localRoot 'llama.stderr.log'
$pidPath = Join-Path $localRoot 'llama.pid'
if (Test-Path $pidPath) {
    $prior = [int](Get-Content $pidPath)
    if (Get-Process -Id $prior -ErrorAction SilentlyContinue) {
        throw 'Prior model-server PID exists; reconcile first'
    }
}
$flags = @('--model', $model, '--alias', 'vera-dolphin-glm24b-iq2xs',
           '--device', 'Vulkan1', '--n-gpu-layers', '14',
           '--ctx-size', '2048', '--threads', '6', '--threads-batch', '6',
           '--batch-size', '128', '--ubatch-size', '64', '--parallel', '1',
           '--host', '127.0.0.1', '--port', '8097',
           '--api-key-file', $keyFile)
$process = Start-Process -FilePath $server.Source -ArgumentList $flags -PassThru -RedirectStandardOutput $logOut -RedirectStandardError $logErr
$process.Id | Set-Content $pidPath -Encoding ascii
Write-Output ('MODEL_SERVER_PID=' + $process.Id)
Write-Output ('MODEL_ALIAS=vera-dolphin-glm24b-iq2xs')
Write-Output ('HTTP_BASE=http://127.0.0.1:8097/v1')
Write-Output ('LOCAL_LOGS=' + $localRoot)
Write-Output 'STARTED_NOT_YET_BEHAVIOR_VERIFIED'
