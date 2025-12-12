<#
CodeEvolve Windows runner.
Fill in your problem name (or pass -ProblemName) and this script will point
CodeEvolve at the correct input, config, and output folders.
#>
param(
    [string]$ProblemName = "",
    [string]$LoadCkpt = "-1",
    [string]$CpuList = "",
    [string]$ConfigChoice = "",
    [string]$RunName = ""
)

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot ".." )).Path
$ProblemsRoot = Join-Path $RepoRoot "problems"
$AvailableProblems = Get-ChildItem -Path $ProblemsRoot -Directory -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Name | Sort-Object

if (-not $AvailableProblems) {
    Write-Error "No problems found in $ProblemsRoot"
    exit 1
}

if (-not $ProblemName) {
    Write-Host "Available problems:"
    foreach ($p in $AvailableProblems) {
        Write-Host "  - $p"
    }
    $DefaultProblem = $AvailableProblems[0]
    $ProblemName = Read-Host "Choose problem [$DefaultProblem]"
    if (-not $ProblemName) { $ProblemName = $DefaultProblem }
}

$BaseDir = Join-Path $RepoRoot (Join-Path "problems" $ProblemName)
$InputDir = Join-Path $BaseDir "input"
$ConfigDir = Join-Path $BaseDir "configs"
$ConfigPath = ""
$DefaultRunName = (Get-Date).ToString('yyyyMMdd_HHmmss')
if (-not $RunName) {
    $RunName = Read-Host "Run name under experiments/$ProblemName [$DefaultRunName]"
    if (-not $RunName) { $RunName = $DefaultRunName }
}
$OutputDir = Join-Path $RepoRoot (Join-Path "experiments" (Join-Path $ProblemName $RunName))
$ApiKeys = @{}

Write-Host "`nOptional: set API key env vars for this run (stored only in memory)."
while ($true) {
    $ApiKeyName = Read-Host "API key env var name (e.g., OPENAI_API_KEY) [press ENTER to skip]"
    if (-not $ApiKeyName) { break }
    $SecureValue = Read-Host "Value for $ApiKeyName" -AsSecureString
    $Ptr = [System.Runtime.InteropServices.Marshal]::SecureStringToGlobalAllocUnicode($SecureValue)
    try {
        $PlainValue = [System.Runtime.InteropServices.Marshal]::PtrToStringUni($Ptr)
    } finally {
        [System.Runtime.InteropServices.Marshal]::ZeroFreeGlobalAllocUnicode($Ptr)
    }
    if (-not $PlainValue) {
        Write-Host "Skipped empty value for $ApiKeyName"
        continue
    }
    [System.Environment]::SetEnvironmentVariable($ApiKeyName, $PlainValue, "Process")
    $ApiKeys[$ApiKeyName] = $true
}

if (-not (Test-Path $InputDir)) {
    Write-Error "Input directory not found: $InputDir"
    exit 1
}

if (-not (Test-Path $ConfigDir)) {
    Write-Error "Config directory not found: $ConfigDir"
    exit 1
}

$AvailableConfigs = Get-ChildItem -Path $ConfigDir -File -Include *.yml, *.yaml, *.json -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Name | Sort-Object

function Set-ConfigFromChoice {
    param([string]$Choice)
    if ($Choice -match '^[0-9]+$') {
        $idx = [int]$Choice - 1
        if ($idx -ge 0 -and $idx -lt $AvailableConfigs.Count) {
            $script:ConfigPath = Join-Path $ConfigDir $AvailableConfigs[$idx]
            return $true
        }
    } elseif ($Choice) {
        $candidate = Join-Path $ConfigDir $Choice
        if (Test-Path $candidate -PathType Leaf) {
            $script:ConfigPath = $candidate
            return $true
        }
    }
    return $false
}

if ($AvailableConfigs.Count -gt 0) {
    Write-Host "Available configs in ${ConfigDir}:"
    for ($i = 0; $i -lt $AvailableConfigs.Count; $i++) {
        $slot = $i + 1
        Write-Host "  [$slot] $($AvailableConfigs[$i])"
    }
    Write-Host "  [N] Provide another config file to copy here"
    $DefaultChoice = "1"
    if (-not $ConfigChoice) {
        $ConfigChoice = Read-Host "Choose config [$DefaultChoice]"
        if (-not $ConfigChoice) { $ConfigChoice = $DefaultChoice }
    } else {
        Write-Host "Using requested config selector: $ConfigChoice"
    }
    if (-not (Set-ConfigFromChoice -Choice $ConfigChoice)) {
        if ($ConfigChoice.ToLower() -ne "n") {
            Write-Error "Invalid choice: $ConfigChoice"
            exit 1
        }
    }
}

if (-not $ConfigPath) {
    if ($ConfigChoice -and (Test-Path $ConfigChoice -PathType Leaf)) {
        $CustomConfig = $ConfigChoice
        Write-Host "Copying requested config file: $CustomConfig"
    } else {
        $CustomConfig = Read-Host "Path to config to copy into $ConfigDir"
    }
    if (-not $CustomConfig) {
        Write-Error "No config provided"
        exit 1
    }
    if (-not (Test-Path $CustomConfig)) {
        Write-Error "Config file not found: $CustomConfig"
        exit 1
    }

    $CustomConfigAbs = [System.IO.Path]::GetFullPath((Resolve-Path -LiteralPath $CustomConfig))
    $DefaultName = [System.IO.Path]::GetFileName($CustomConfigAbs)
    $CustomName = Read-Host "Save as [$DefaultName]"
    if (-not $CustomName) { $CustomName = $DefaultName }
    $ConfigPath = Join-Path $ConfigDir $CustomName
    Copy-Item -LiteralPath $CustomConfigAbs -Destination $ConfigPath -Force
    Write-Host "Copied custom config to: $ConfigPath"
}

if (-not (Test-Path $ConfigPath)) {
    Write-Error "Config file not found: $ConfigPath"
    exit 1
}

if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
}

if (-not (Get-Command codeevolve -ErrorAction SilentlyContinue)) {
    Write-Error "'codeevolve' CLI not found in PATH. Activate your env first: conda activate codeevolve"
    exit 1
}

Write-Host "➡️  Using problem: $ProblemName"
Write-Host "   Input:  $InputDir"
Write-Host "   Config: $ConfigPath"
Write-Host "   Output: $OutputDir"
Write-Host "`nTip: conda activate codeevolve  # ensure the environment is ready"

$command = @(
    "codeevolve",
    "--inpt_dir=$InputDir",
    "--cfg_path=$ConfigPath",
    "--out_dir=$OutputDir",
    "--load_ckpt=$LoadCkpt",
    "--terminal_logging"
)

if ($CpuList -ne "") {
    Write-Warning "CPU pinning is not set on Windows by default; set $env:OMP_NUM_THREADS or similar if needed."
}

$process = & $command[0] $command[1..($command.Length-1)]
$status = $LASTEXITCODE

if ($ApiKeys.Keys.Count -gt 0) {
    Write-Host "Cleaning up API key variables..."
    foreach ($key in $ApiKeys.Keys) {
        [System.Environment]::SetEnvironmentVariable($key, $null, "Process")
    }
}

exit $status
