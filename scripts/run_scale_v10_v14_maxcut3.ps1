param(
    [string]$Device = "cuda",
    [string]$OutputDir = "outputs/scale_v10_v14_maxcut3_full",
    [int]$ClassicalTimeLimitSeconds = 3600
)

$ErrorActionPreference = "Stop"
$Python = ".venv\Scripts\python.exe"

$CudaAvailable = (& $Python -c "import torch; print(torch.cuda.is_available())").Trim()
if ($Device -eq "cuda" -and $CudaAvailable -ne "True") {
    Write-Warning "Device was set to cuda, but this Python environment cannot see CUDA. The run will fall back to CPU unless CUDA-enabled PyTorch is installed."
}

& $Python classical\scale_v10_v14_maxcut3.py `
    --min-n 512 `
    --max-n 4096 `
    --size-mode doubling `
    --seeds 0 1 2 3 4 5 6 7 8 9 `
    --models v10 v14 `
    --device $Device `
    --output-dir $OutputDir `
    --adaptive-refine `
    --refine-step-n 128 `
    --threshold-metric C_d `
    --classical-time-limit-seconds $ClassicalTimeLimitSeconds `
    --gw-rank 64 `
    --gw-steps 1200 `
    --gw-restarts 2 `
    --gw-rounding-samples 4096 `
    --random-flip-samples 1024 `
    --random-flip-batch-size 256 `
    --greedy-restarts 32 `
    --greedy-passes 220 `
    --sample-count 256 `
    --v10-rounds 100 `
    --v10-epochs 200 `
    --v10-symmetry-trials 4 `
    --v14-rounds 280 `
    --v14-epochs 110 `
    --v14-head-count 1
