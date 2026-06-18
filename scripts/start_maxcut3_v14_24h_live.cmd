@echo off
setlocal

set "ROOT=%~dp0.."
cd /d "%ROOT%"

set "OUT=outputs\maxcut3_v14_24h_research_live"
set "BASE=%OUT%\baselines"
if not exist "%OUT%" mkdir "%OUT%"
if not exist "%BASE%" mkdir "%BASE%"

set "STDOUT=%OUT%\research_loop_stdout_cmd.log"
set "STDERR=%OUT%\research_loop_stderr_cmd.log"

".venv\Scripts\python.exe" scripts\run_maxcut3_v14_research_loop.py ^
  --output-dir "%OUT%" ^
  --baseline-dir "%BASE%" ^
  --device cuda ^
  --hours 24 ^
  --max-runs-per-cycle 6 ^
  --screen-rounds 140 ^
  --screen-epochs 55 ^
  --exploit-rounds 260 ^
  --exploit-epochs 115 ^
  --num-samples 384 ^
  --local-search-passes 240 ^
  --sample-local-search-passes 120 ^
  --skip-baseline ^
  --resume ^
  1>> "%STDOUT%" ^
  2>> "%STDERR%"

echo EXITCODE %ERRORLEVEL%>> "%STDERR%"
