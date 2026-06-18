$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$outputDir = Join-Path $root "outputs\maxcut3_v14_24h_research_live"
$baselineDir = Join-Path $outputDir "baselines"
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
New-Item -ItemType Directory -Force -Path $baselineDir | Out-Null

$stdout = Join-Path $outputDir "research_loop_stdout.log"
$stderr = Join-Path $outputDir "research_loop_stderr.log"
$python = Join-Path $root ".venv\Scripts\python.exe"

& $python `
  "scripts\run_maxcut3_v14_research_loop.py" `
  "--output-dir" "outputs\maxcut3_v14_24h_research_live" `
  "--baseline-dir" "outputs\maxcut3_v14_24h_research_live\baselines" `
  "--device" "cuda" `
  "--hours" "24" `
  "--max-runs-per-cycle" "6" `
  "--screen-rounds" "140" `
  "--screen-epochs" "55" `
  "--exploit-rounds" "260" `
  "--exploit-epochs" "115" `
  "--num-samples" "384" `
  "--local-search-passes" "240" `
  "--sample-local-search-passes" "120" `
  "--skip-baseline" `
  1> $stdout `
  2> $stderr
