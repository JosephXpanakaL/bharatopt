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

$candidates = @(
  Join-Path (Get-Location) "build\bin\bharatopt_cli.exe",
  Join-Path (Get-Location) "build\bin\bharatopt.exe",
  Join-Path (Get-Location) "build\bharatopt_cli.exe",
  Join-Path (Get-Location) "build\bharatopt.exe",
  Join-Path (Get-Location) "build\$buildType\bharatopt_cli.exe",
  Join-Path (Get-Location) "build\$buildType\bharatopt.exe"
)

$exe = $null
foreach ($c in $candidates) {
  if (Test-Path $c) {
    $exe = $c
    break
  }
}

if ($exe) {
  Write-Host "Build complete: $exe"
  if ($Cuda) { & $exe --demo --cuda } else { & $exe --demo }
} else {
  Write-Host "Build finished, but executable not found in expected output directories."
}
