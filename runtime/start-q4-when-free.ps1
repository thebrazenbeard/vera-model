# Guarded high-fidelity Q4_K_M sidecar. Never modifies VeraOS or Portal routes.
param([switch]$InspectOnly)
$ErrorActionPreference = 'Stop'
$modelRoot = 'D:\VERA\models\gguf\vera-glm24b-q4-k-m'
$name = 'Dolphin-Mistral-GLM-4.7-Flash-24B-Venice-Edition-Thinking-Uncensored.i1-Q4_K_M.gguf'
$model = Join-Path $modelRoot $name
$expectedBytes = [int64]14333910944
$expectedSha = 'cb3b3b986bcecac5a2f3f1aa814a2a1e2a8b430bfeaae7bdab50c6ecfed7511a'
$minimumRamGiB = 12.0
$minimumFreeVramMiB = 2800
$os = Get-CimInstance Win32_OperatingSystem
$freeGiB = [math]::Round([double]$os.FreePhysicalMemory / 1048576.0, 2)
$gpu = & nvidia-smi --query-gpu=memory.total,memory.free --format=csv,noheader,nounits
if ($LASTEXITCODE -ne 0 -or @($gpu).Count -ne 1) {
    throw 'Unable to verify available NVIDIA VRAM'
}
$parts = $gpu.Trim().Split(',')
$totalGpu = [int]$parts[0].Trim()
$freeGpu = [int]$parts[1].Trim()
Write-Output ('RAM_FREE_GIB=' + $freeGiB)
Write-Output ('GPU_FREE_MIB=' + $freeGpu)
Write-Output 'GGUF_QUANT=Q4_K_M'
Write-Output ('MIN_RAM_FREE_GIB=' + $minimumRamGiB)
Write-Output ('MIN_GPU_FREE_MIB=' + $minimumFreeVramMiB)
if ((-not (Test-Path -LiteralPath $model -PathType Leaf)) -or
    (Get-Item -LiteralPath $model).Length -ne $expectedBytes) {
    Write-Output 'HOLD: exact Q4 donor GGUF is unavailable'
    exit 2
}
if ($freeGiB -lt $minimumRamGiB -or $freeGpu -lt $minimumFreeVramMiB) {
    Write-Output 'HOLD: leave existing Lappy workloads intact; no high-fidelity model started'
    exit 2
}
if ($InspectOnly) {
    Write-Output 'CAPACITY_GATE_PASSED_ONLY_NOT_STARTED'
    exit 0
}
$occupied = Get-NetTCPConnection -State Listen -LocalPort 8098 -ErrorAction SilentlyContinue
if ($occupied) { throw 'Q4 port 8098 already occupied' }
$work = Join-Path $env:LOCALAPPDATA 'VeraModel\glm24b-q4'
if (-not (Test-Path $work)) {
    New-Item -ItemType Directory -Path $work | Out-Null
}
$pidFile = Join-Path $work 'llama.pid'
if (Test-Path $pidFile) {
    $old = Get-Process -Id ([int](Get-Content $pidFile)) -ErrorAction SilentlyContinue
    if ($old) { throw 'Previously recorded Q4 server PID is active' }
}
$authKey = Join-Path $env:LOCALAPPDATA 'VeraModel\glm24b\api-key.txt'
if (-not (Test-Path $authKey)) { throw 'Local API key file unavailable' }
$observed = (Get-FileHash -LiteralPath $model -Algorithm SHA256).Hash.ToLowerInvariant()
if ($observed -ne $expectedSha) { throw 'Q4 digest mismatch' }
$llama = Get-Command llama-server.exe -ErrorAction Stop
$argsList = @('--model', $model, '--alias', 'vera-dolphin-glm24b-q4km',
              '--device', 'Vulkan1', '--n-gpu-layers', '7',
              '--ctx-size', '2048', '--threads', '6', '--threads-batch', '6',
              '--batch-size', '128', '--ubatch-size', '64', '--parallel', '1',
              '--host', '127.0.0.1', '--port', '8098',
              '--api-key-file', $authKey)
$server = Start-Process -FilePath $llama.Source -ArgumentList $argsList -PassThru -RedirectStandardOutput (Join-Path $work 'llama.stdout.log') -RedirectStandardError (Join-Path $work 'llama.stderr.log')
$server.Id | Set-Content $pidFile -Encoding ascii
Write-Output ('Q4_MODEL_SERVER_PID=' + $server.Id)
Write-Output 'Q4_LAUNCHED_NOT_YET_VERIFIED'
