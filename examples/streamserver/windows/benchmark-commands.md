# Benchmark Commands for Official Releases

This file contains the exact commands to run each official release version and measure their performance.

## Prerequisites

Make sure you're in the whisper.cpp repository root directory:
```powershell
cd C:\Users\at15\source\repos\whisper.cpp
```

## Release Locations

Based on your Downloads folder:
- **Default**: `C:\Users\at15\Downloads\whisper-bin-x64\Release\whisper-cli.exe`
- **OpenBLAS**: `C:\Users\at15\Downloads\whisper-blas-bin-x64\Release\whisper-cli.exe`
- **CUDA**: `C:\Users\at15\Downloads\whisper-cublas-12.4.0-bin-x64\Release\whisper-cli.exe`

## Benchmark Commands

### 1. Default Version (CPU only)

```powershell
$DefaultCli = "C:\Users\at15\Downloads\whisper-bin-x64\Release\whisper-cli.exe"
Measure-Command {
    & $DefaultCli --file samples\recording-zh-xiaogao-jie.wav --model models\ggml-medium.bin --language zh --vad --vad-model models\ggml-silero-v6.2.0.bin --output-srt
}
```

**To capture output with timing info:**
```powershell
$DefaultCli = "C:\Users\at15\Downloads\whisper-bin-x64\Release\whisper-cli.exe"
$output = & $DefaultCli --file samples\recording-zh-xiaogao-jie.wav --model models\ggml-medium.bin --language zh --vad --vad-model models\ggml-silero-v6.2.0.bin --output-srt 2>&1 | Out-String
$output | Out-File "default-output.txt"
$output
```

### 2. OpenBLAS Version

```powershell
$BlasCli = "C:\Users\at15\Downloads\whisper-blas-bin-x64\Release\whisper-cli.exe"
Measure-Command {
    & $BlasCli --file samples\recording-zh-xiaogao-jie.wav --model models\ggml-medium.bin --language zh --vad --vad-model models\ggml-silero-v6.2.0.bin --output-srt
}
```

**To capture output with timing info:**
```powershell
$BlasCli = "C:\Users\at15\Downloads\whisper-blas-bin-x64\Release\whisper-cli.exe"
$output = & $BlasCli --file samples\recording-zh-xiaogao-jie.wav --model models\ggml-medium.bin --language zh --vad --vad-model models\ggml-silero-v6.2.0.bin --output-srt 2>&1 | Out-String
$output | Out-File "blas-output.txt"
$output
```

### 3. CUDA Version

```powershell
$CudaCli = "C:\Users\at15\Downloads\whisper-cublas-12.4.0-bin-x64\Release\whisper-cli.exe"
Measure-Command {
    & $CudaCli --file samples\recording-zh-xiaogao-jie.wav --model models\ggml-medium.bin --language zh --vad --vad-model models\ggml-silero-v6.2.0.bin --output-srt
}
```

**To capture output with timing info:**
```powershell
$CudaCli = "C:\Users\at15\Downloads\whisper-cublas-12.4.0-bin-x64\Release\whisper-cli.exe"
$output = & $CudaCli --file samples\recording-zh-xiaogao-jie.wav --model models\ggml-medium.bin --language zh --vad --vad-model models\ggml-silero-v6.2.0.bin --output-srt 2>&1 | Out-String
$output | Out-File "cuda-output.txt"
$output
```

## Quick Comparison Script

Run all three versions and save results:

```powershell
# Set paths
$DefaultCli = "C:\Users\at15\Downloads\whisper-bin-x64\Release\whisper-cli.exe"
$BlasCli = "C:\Users\at15\Downloads\whisper-blas-bin-x64\Release\whisper-cli.exe"
$CudaCli = "C:\Users\at15\Downloads\whisper-cublas-12.4.0-bin-x64\Release\whisper-cli.exe"

# Common arguments
$args = "--file samples\recording-zh-xiaogao-jie.wav --model models\ggml-medium.bin --language zh --vad --vad-model models\ggml-silero-v6.2.0.bin --output-srt"

# Test Default
Write-Host "`n=== Testing DEFAULT version ===" -ForegroundColor Yellow
$defaultTime = Measure-Command { & $DefaultCli $args.Split(' ') }
Write-Host "Default: $($defaultTime.TotalSeconds) seconds" -ForegroundColor Green

# Test OpenBLAS
Write-Host "`n=== Testing OpenBLAS version ===" -ForegroundColor Yellow
$blasTime = Measure-Command { & $BlasCli $args.Split(' ') }
Write-Host "OpenBLAS: $($blasTime.TotalSeconds) seconds" -ForegroundColor Green

# Test CUDA
Write-Host "`n=== Testing CUDA version ===" -ForegroundColor Yellow
$cudaTime = Measure-Command { & $CudaCli $args.Split(' ') }
Write-Host "CUDA: $($cudaTime.TotalSeconds) seconds" -ForegroundColor Green

# Summary
Write-Host "`n=== Summary ===" -ForegroundColor Cyan
Write-Host "Default:  $($defaultTime.TotalSeconds)s"
Write-Host "OpenBLAS: $($blasTime.TotalSeconds)s"
Write-Host "CUDA:     $($cudaTime.TotalSeconds)s"
```

## Extracting Timing Information

After running each command, look for these patterns in the output:

### VAD Timing
Look for lines like:
```
whisper_vad_detect_speech: vad time = 563.09 ms processing 8064 samples
```

### INFER Timing
Look for lines like:
```
INFER session=... DONE time=3189ms text_len=33
```

### Extract VAD times from output file:
```powershell
$content = Get-Content "default-output.txt" -Raw
$vadTimes = [regex]::Matches($content, "vad time = ([\d.]+) ms")
$vadTimes | ForEach-Object { Write-Host "VAD: $($_.Groups[1].Value) ms" }
```

### Extract INFER times from output file:
```powershell
$content = Get-Content "default-output.txt" -Raw
$inferTimes = [regex]::Matches($content, "INFER.*DONE time=(\d+)ms")
$inferTimes | ForEach-Object { Write-Host "INFER: $($_.Groups[1].Value) ms" }
```

## Notes

- The `Measure-Command` cmdlet will show the total execution time
- The whisper-cli output contains detailed VAD and INFER timing information
- Save the output to files for detailed analysis
- Run each test multiple times and average for more accurate results
- Make sure no other processes are using significant CPU/GPU during testing

