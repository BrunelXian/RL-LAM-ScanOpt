Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = "D:\Projects\RL-LAM-ScanOpt"
$Runner = Join-Path $RepoRoot "scripts\abaqus\run_remaining_6_abaqus_jobs.py"
$SummaryDir = Join-Path $RepoRoot "abaqus-models\night_run_remaining_6_summary"
$StdOutPath = Join-Path $SummaryDir "runner_stdout.log"
$StdErrPath = Join-Path $SummaryDir "runner_stderr.log"

if (-not (Test-Path -LiteralPath $Runner -PathType Leaf)) {
    throw "Python runner not found: $Runner"
}

if (-not (Test-Path -LiteralPath $SummaryDir -PathType Container)) {
    New-Item -ItemType Directory -Force -Path $SummaryDir | Out-Null
}

$ArgumentList = @(
    "`"$Runner`"",
    "--cpus", "12",
    "--continue-on-failure", "false"
)

$process = Start-Process `
    -FilePath "python" `
    -ArgumentList $ArgumentList `
    -WorkingDirectory $RepoRoot `
    -RedirectStandardOutput $StdOutPath `
    -RedirectStandardError $StdErrPath `
    -WindowStyle Hidden `
    -PassThru

Write-Host "Started Python Abaqus runner PID: $($process.Id)"
Write-Host "Runner: $Runner"
Write-Host "Summary directory: $SummaryDir"
Write-Host "Stdout: $StdOutPath"
Write-Host "Stderr: $StdErrPath"
