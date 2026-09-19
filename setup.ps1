param(
  [switch]$Cuda,
  [switch]$DebugBuild
)

$ErrorActionPreference = "Stop"
$buildType = if ($DebugBuild) { "Debug" } else { "Release" }
$cudaValue = if ($Cuda) { "ON" } else { "OFF" }

Write-Host "== BharatOpt build =="
cmake -S . -B build -DCMAKE_BUILD_TYPE=$buildType -DBHARATOPT_ENABLE_CUDA=$cudaValue
cmake --build build --config $buildType
ctest --test-dir build --build-config $buildType --output-on-failure

$exe = Join-Path (Get-Location) ("buildharatopt_cli.exe")
if (Test-Path $exe) {
  Write-Host "Build complete: $exe"
  if ($Cuda) { & $exe --demo --cuda } else { & $exe --demo }
}
