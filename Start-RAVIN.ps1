param(
    [switch]$CheckOnly,
    [switch]$SkipBrowser,
    [switch]$ForceSetup,
    [switch]$RefreshPolicyCache
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $RepoRoot

$FrontendDir = Join-Path $RepoRoot "frontend"
$VenvDir = Join-Path $RepoRoot ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"

$ApiUrl = "http://127.0.0.1:8000"
$HealthUrl = "$ApiUrl/health"
$FrontendUrl = "http://localhost:3000"

$RequiredOllamaModel = "qwen3:4b-instruct"

$DemoStateDir = Join-Path $env:TEMP "RAVIN-Demo"
$ProcessStateFile = Join-Path $DemoStateDir "processes.json"

New-Item `
    -ItemType Directory `
    -Path $DemoStateDir `
    -Force |
    Out-Null

function Write-Step {
    param(
        [string]$Message
    )

    Write-Host ""
    Write-Host "=== $Message ==="
}

function Update-ProcessPath {
    $machinePath = [Environment]::GetEnvironmentVariable(
        "Path",
        "Machine"
    )

    $userPath = [Environment]::GetEnvironmentVariable(
        "Path",
        "User"
    )

    $env:Path = "$machinePath;$userPath"
}

function Get-CommandPath {
    param(
        [string]$Name
    )

    $command = Get-Command `
        $Name `
        -ErrorAction SilentlyContinue

    if ($command) {
        return $command.Source
    }

    return $null
}

function Invoke-NativeCommand {
    param(
        [Parameter(Mandatory)]
        [string]$FilePath,

        [Parameter(Mandatory)]
        [string[]]$ArgumentList,

        [Parameter(Mandatory)]
        [string]$FailureMessage
    )

    $previousErrorActionPreference = $ErrorActionPreference
    $exitCode = $null

    try {
        # Native tools commonly write warnings and progress to stderr.
        # Do not treat stderr alone as a PowerShell terminating error.
        # Their process exit code determines success or failure.
        $ErrorActionPreference = "Continue"

        & $FilePath @ArgumentList 2>&1 |
            Out-Host

        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }

    if ($exitCode -ne 0) {
        throw "$FailureMessage Exit code: $exitCode"
    }
}

function Get-WingetPath {
    $winget = Get-CommandPath "winget"

    if (-not $winget) {
        throw @"
Windows Package Manager (winget) is required to automatically install
missing machine prerequisites.

Install or enable winget, then run Start-RAVIN again.
"@
    }

    return $winget
}

function Install-WingetPackage {
    param(
        [string]$PackageId,
        [string]$DisplayName
    )

    $winget = Get-WingetPath

    Write-Host "$DisplayName is missing."
    Write-Host "Installing $DisplayName with winget..."

    Invoke-NativeCommand `
        -FilePath $winget `
        -ArgumentList @(
            "install",
            "--id",
            $PackageId,
            "--exact",
            "--silent",
            "--accept-package-agreements",
            "--accept-source-agreements"
        ) `
        -FailureMessage "Failed to install $DisplayName."

    Update-ProcessPath
}

function Get-Python312 {
    $py = Get-CommandPath "py"

    if ($py) {
        $version = & $py -3.12 -c `
            "import sys; print('.'.join(map(str, sys.version_info[:3])))" `
            2>$null

        if ($LASTEXITCODE -eq 0) {
            return @{
                Launcher = $py
                Prefix = @("-3.12")
                Version = $version.Trim()
            }
        }
    }

    $python = Get-CommandPath "python"

    if ($python) {
        $versionText = & $python -c `
            "import sys; print('.'.join(map(str, sys.version_info[:3])))" `
            2>$null

        if ($LASTEXITCODE -eq 0) {
            $version = [version]$versionText.Trim()

            if (
                $version -ge [version]"3.12.0" -and
                $version -lt [version]"3.13.0"
            ) {
                return @{
                    Launcher = $python
                    Prefix = @()
                    Version = $versionText.Trim()
                }
            }
        }
    }

    $knownPython = Join-Path `
        $env:LOCALAPPDATA `
        "Programs\Python\Python312\python.exe"

    if (Test-Path $knownPython) {
        $versionText = & $knownPython -c `
            "import sys; print('.'.join(map(str, sys.version_info[:3])))"

        return @{
            Launcher = $knownPython
            Prefix = @()
            Version = $versionText.Trim()
        }
    }

    return $null
}

function Initialize-PythonRuntime {
    Write-Step "PYTHON 3.12"

    $python = Get-Python312

    if (-not $python) {
        Install-WingetPackage `
            -PackageId "Python.Python.3.12" `
            -DisplayName "Python 3.12"

        $python = Get-Python312
    }

    if (-not $python) {
        throw "Python 3.12 could not be located after installation."
    }

    Write-Host "Python $($python.Version) ready."

    return $python
}

function Initialize-VirtualEnvironment {
    param(
        [hashtable]$Python
    )

    Write-Step "PYTHON VIRTUAL ENVIRONMENT"

    if (-not (Test-Path $VenvPython)) {
        Write-Host "Creating .venv..."

        if ($Python.Prefix.Count -gt 0) {
            & $Python.Launcher `
                $Python.Prefix `
                -m venv `
                $VenvDir
        }
        else {
            & $Python.Launcher `
                -m venv `
                $VenvDir
        }

        if ($LASTEXITCODE -ne 0) {
            throw "Failed to create the Python virtual environment."
        }
    }
    else {
        Write-Host ".venv already exists."
    }

    if (-not (Test-Path $VenvPython)) {
        throw "Virtual environment Python executable was not created."
    }

    Write-Host "Virtual environment ready."
}

function Sync-PythonDependencies {
    Write-Step "PYTHON DEPENDENCIES"

    $requirements = @(
        (Join-Path $RepoRoot "requirements.txt"),
        (Join-Path $RepoRoot "requirements-api.txt")
    )

    foreach ($file in $requirements) {
        if (-not (Test-Path $file)) {
            throw "Required dependency file is missing: $file"
        }
    }

    $requirementState = (
        $requirements |
        ForEach-Object {
            $hash = (
                Get-FileHash `
                    -Path $_ `
                    -Algorithm SHA256
            ).Hash

            "$(Split-Path $_ -Leaf):$hash"
        }
    ) -join "`n"

    $stamp = Join-Path `
        $VenvDir `
        ".ravin-requirements.sha256"

    $installedState = ""

    if (Test-Path $stamp) {
        $installedState = Get-Content `
            $stamp `
            -Raw
    }

    if (
        $ForceSetup -or
        $installedState.Trim() -ne $requirementState.Trim()
    ) {
        Write-Host "Installing/updating Python packages..."

        Invoke-NativeCommand `
            -FilePath $VenvPython `
            -ArgumentList @(
                "-m",
                "pip",
                "install",
                "--upgrade",
                "pip"
            ) `
            -FailureMessage "Failed to prepare pip."

        if ($LASTEXITCODE -ne 0) {
            throw "Failed to prepare pip."
        }

        foreach ($file in $requirements) {
            Invoke-NativeCommand `
                -FilePath $VenvPython `
                -ArgumentList @(
                    "-m",
                    "pip",
                    "install",
                    "-r",
                    $file
                ) `
                -FailureMessage "Python dependency installation failed for $file."
        }

        Set-Content `
            -Path $stamp `
            -Value $requirementState `
            -Encoding UTF8

        Write-Host "Python dependencies installed."
    }
    else {
        Write-Host "Python dependency manifest unchanged."
        Write-Host "Existing environment will be reused."
    }

    Invoke-NativeCommand `
        -FilePath $VenvPython `
        -ArgumentList @(
            "-m",
            "pip",
            "check"
        ) `
        -FailureMessage "pip check reported an invalid Python environment."

    Write-Host "Python dependency check passed."
}

function Initialize-NodeRuntime {
    Write-Step "NODE.JS AND NPM"

    $node = Get-CommandPath "node"

    $validNode = $false

    if ($node) {
        $rawVersion = (& $node --version).Trim()
        $nodeVersion = [version]$rawVersion.TrimStart("v")

        if ($nodeVersion -ge [version]"20.9.0") {
            $validNode = $true
            Write-Host "Node.js $rawVersion ready."
        }
    }

    if (-not $validNode) {
        Install-WingetPackage `
            -PackageId "OpenJS.NodeJS.LTS" `
            -DisplayName "Node.js LTS"

        $node = Get-CommandPath "node"

        if (-not $node) {
            throw "Node.js could not be located after installation."
        }

        $rawVersion = (& $node --version).Trim()
        $nodeVersion = [version]$rawVersion.TrimStart("v")

        if ($nodeVersion -lt [version]"20.9.0") {
            throw "Node.js 20.9.0 or newer is required."
        }

        Write-Host "Node.js $rawVersion ready."
    }

    $npm = Get-CommandPath "npm"

    if (-not $npm) {
        throw "npm is missing even though Node.js is installed."
    }

    $npmVersion = (& $npm --version).Trim()

    Write-Host "npm $npmVersion ready."

    return $npm
}

function Sync-FrontendDependencies {
    param(
        [string]$Npm
    )

    Write-Step "FRONTEND DEPENDENCIES"

    $packageJson = Join-Path `
        $FrontendDir `
        "package.json"

    $packageLock = Join-Path `
        $FrontendDir `
        "package-lock.json"

    $nodeModules = Join-Path `
        $FrontendDir `
        "node_modules"

    if (-not (Test-Path $packageJson)) {
        throw "frontend/package.json is missing."
    }

    if (-not (Test-Path $packageLock)) {
        throw "frontend/package-lock.json is missing."
    }

    $lockHash = (
        Get-FileHash `
            -Path $packageLock `
            -Algorithm SHA256
    ).Hash

    $stamp = Join-Path `
        $nodeModules `
        ".ravin-package-lock.sha256"

    $installedHash = ""

    if (Test-Path $stamp) {
        $installedHash = (
            Get-Content `
                $stamp `
                -Raw
        ).Trim()
    }

    if (
        $ForceSetup -or
        -not (Test-Path $nodeModules) -or
        $installedHash -ne $lockHash
    ) {
        Write-Host "Installing exact frontend dependencies with npm ci..."

        Push-Location $FrontendDir

        try {
            Invoke-NativeCommand `
                -FilePath $Npm `
                -ArgumentList @(
                    "ci"
                ) `
                -FailureMessage "npm ci failed."
        }
        finally {
            Pop-Location
        }

        Set-Content `
            -Path $stamp `
            -Value $lockHash `
            -Encoding UTF8

        Write-Host "Frontend dependencies installed."
    }
    else {
        Write-Host "package-lock.json unchanged."
        Write-Host "Existing frontend dependencies will be reused."
    }
}

function Get-OllamaPath {
    $ollama = Get-CommandPath "ollama"

    if ($ollama) {
        return $ollama
    }

    $knownOllama = Join-Path `
        $env:LOCALAPPDATA `
        "Programs\Ollama\ollama.exe"

    if (Test-Path $knownOllama) {
        return $knownOllama
    }

    return $null
}

function Test-OllamaApi {
    try {
        Invoke-RestMethod `
            -Uri "http://127.0.0.1:11434/api/tags" `
            -Method Get `
            -TimeoutSec 3 |
            Out-Null

        return $true
    }
    catch {
        return $false
    }
}

function Wait-ForOllama {
    param(
        [int]$TimeoutSeconds = 30
    )

    $deadline = (
        Get-Date
    ).AddSeconds($TimeoutSeconds)

    while ((Get-Date) -lt $deadline) {
        if (Test-OllamaApi) {
            return
        }

        Start-Sleep -Seconds 1
    }

    throw "Ollama did not become ready within $TimeoutSeconds seconds."
}

function Initialize-OllamaRuntime {
    Write-Step "OLLAMA"

    $ollama = Get-OllamaPath

    if (-not $ollama) {
        Install-WingetPackage `
            -PackageId "Ollama.Ollama" `
            -DisplayName "Ollama"

        $ollama = Get-OllamaPath
    }

    if (-not $ollama) {
        throw "Ollama could not be located after installation."
    }

    $version = & $ollama --version 2>&1

    Write-Host "$version"

    if (-not (Test-OllamaApi)) {
        Write-Host "Starting Ollama service..."

        Start-Process `
            -FilePath $ollama `
            -ArgumentList "serve" `
            -WindowStyle Minimized |
            Out-Null

        Wait-ForOllama
    }

    Write-Host "Ollama API ready."

    $models = & $ollama list 2>&1 |
        Out-String

    if (
        $models -notmatch
        [regex]::Escape($RequiredOllamaModel)
    ) {
        Write-Host "Required model is missing."
        Write-Host "Pulling $RequiredOllamaModel..."

        Invoke-NativeCommand `
            -FilePath $ollama `
            -ArgumentList @(
                "pull",
                $RequiredOllamaModel
            ) `
            -FailureMessage "Failed to pull $RequiredOllamaModel."
    }

    Write-Host "$RequiredOllamaModel ready."

    return $ollama
}

function Test-PortListening {
    param(
        [int]$Port
    )

    $listener = Get-NetTCPConnection `
        -LocalPort $Port `
        -State Listen `
        -ErrorAction SilentlyContinue

    return $null -ne $listener
}

function Test-RavinReady {
    try {
        $health = Invoke-RestMethod `
            -Uri $HealthUrl `
            -Method Get `
            -TimeoutSec 5

        return (
            $health.status -eq "ok" -and
            $health.service_ready -eq $true
        )
    }
    catch {
        return $false
    }
}

function Wait-ForRavin {
    param(
        [int]$TimeoutSeconds = 300
    )

    Write-Host "Waiting for RAVIN backend readiness..."

    $deadline = (
        Get-Date
    ).AddSeconds($TimeoutSeconds)

    while ((Get-Date) -lt $deadline) {
        if (Test-RavinReady) {
            Write-Host "RAVIN backend is ready."
            return
        }

        Start-Sleep -Seconds 2
    }

    throw @"
RAVIN did not report service_ready=true within $TimeoutSeconds seconds.
Review the backend terminal for the startup error.
"@
}

function Test-FrontendReady {
    try {
        $response = Invoke-WebRequest `
            -Uri $FrontendUrl `
            -UseBasicParsing `
            -TimeoutSec 5

        return (
            $response.StatusCode -ge 200 -and
            $response.StatusCode -lt 500
        )
    }
    catch {
        return $false
    }
}

function Wait-ForFrontend {
    param(
        [int]$TimeoutSeconds = 120
    )

    Write-Host "Waiting for RAVIN frontend..."

    $deadline = (
        Get-Date
    ).AddSeconds($TimeoutSeconds)

    while ((Get-Date) -lt $deadline) {
        if (Test-FrontendReady) {
            Write-Host "RAVIN frontend is ready."
            return
        }

        Start-Sleep -Seconds 1
    }

    throw @"
RAVIN frontend did not become ready within $TimeoutSeconds seconds.
Review the frontend terminal for the startup error.
"@
}

function Invoke-FrontendBuild {
    param(
        [string]$Npm
    )

    Write-Step "PRODUCTION FRONTEND BUILD"

    Push-Location $FrontendDir

    try {
        Invoke-NativeCommand `
            -FilePath $Npm `
            -ArgumentList @(
                "run",
                "build"
            ) `
            -FailureMessage "Frontend production build failed."
    }
    finally {
        Pop-Location
    }

    Write-Host "Frontend production build passed."
}

function Start-RavinBackend {
    param(
        [switch]$RefreshPolicyCache
    )

    Write-Step "START RAVIN BACKEND"

    if (Test-PortListening 8000) {
        if ($RefreshPolicyCache) {
            throw @"
A policy-cache refresh was requested, but the RAVIN backend is already
running on port 8000.

Stop the existing backend before requesting a full cache refresh.
"@
        }

        if (Test-RavinReady) {
            Write-Host "RAVIN backend is already running."
            return $null
        }

        throw @"
Port 8000 is already in use by another process and RAVIN health
verification failed.
"@
    }

    $refreshEnvironment = ""

    if ($RefreshPolicyCache) {
        Write-Host "Full policy-cache refresh requested."
        Write-Host "This may take approximately 10-15 minutes."

        $refreshEnvironment = @'
$env:RAVIN_REFRESH_POLICY_CACHE = "1"
'@
    }
    else {
        Write-Host "Normal startup requested."
        Write-Host "A validated policy cache will be used when available."
    }

    $command = @"
$refreshEnvironment
& '$VenvPython' -m uvicorn backend.api.main:app --host 127.0.0.1 --port 8000
"@

    $process = Start-Process `
        -FilePath "powershell.exe" `
        -ArgumentList @(
            "-NoExit",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            $command
        ) `
        -WorkingDirectory $RepoRoot `
        -PassThru

    Write-Host "Backend terminal started (PID $($process.Id))."

    $readinessTimeout = 1200

    Wait-ForRavin `
        -TimeoutSeconds $readinessTimeout

    return $process
}

function Start-RavinFrontend {
    param(
        [string]$Npm
    )

    Write-Step "START RAVIN FRONTEND"

    if (Test-PortListening 3000) {
        if (Test-FrontendReady) {
            Write-Host "RAVIN frontend is already running."
            return $null
        }

        throw @"
Port 3000 is already in use by another process and the RAVIN frontend
could not be verified.
"@
    }

    $command = @"
Set-Location -LiteralPath '$FrontendDir'
& '$Npm' run start
"@

    $process = Start-Process `
        -FilePath "powershell.exe" `
        -ArgumentList @(
            "-NoExit",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            $command
        ) `
        -WorkingDirectory $FrontendDir `
        -PassThru

    Write-Host "Frontend terminal started (PID $($process.Id))."

    Wait-ForFrontend

    return $process
}

try {
    Write-Host "RAVIN v1.0.0 Demo Environment"
    Write-Host "Repository: $RepoRoot"

    Write-Step "ENVIRONMENT BOOTSTRAP"

    $python = Initialize-PythonRuntime

    Initialize-VirtualEnvironment `
        -Python $python

    Sync-PythonDependencies

    $npm = Initialize-NodeRuntime

    Sync-FrontendDependencies `
        -Npm $npm

    $ollama = Initialize-OllamaRuntime

    Write-Step "ENVIRONMENT CHECK COMPLETE"

    Write-Host "Python environment: READY"
    Write-Host "Frontend dependencies: READY"
    Write-Host "Ollama: READY"
    Write-Host "${RequiredOllamaModel}: READY"

    if ($CheckOnly) {
        Write-Host ""
        Write-Host "CheckOnly requested."
        Write-Host "RAVIN services were not started."
        exit 0
    }

    Invoke-FrontendBuild `
        -Npm $npm

    $backend = Start-RavinBackend `
        -RefreshPolicyCache:$RefreshPolicyCache

    $frontend = Start-RavinFrontend `
        -Npm $npm

    $state = @{
        backend_pid = if ($backend) {
            $backend.Id
        }
        else {
            $null
        }

        frontend_pid = if ($frontend) {
            $frontend.Id
        }
        else {
            $null
        }

        started_at = (
            Get-Date
        ).ToString("o")

        version = "1.0.0"
    }

    $state |
        ConvertTo-Json |
        Set-Content `
            -Path $ProcessStateFile `
            -Encoding UTF8

    Write-Step "RAVIN DEMO READY"

    Write-Host "Frontend: $FrontendUrl"
    Write-Host "API:      $ApiUrl"
    Write-Host "Version:  v1.0.0"

    if (-not $SkipBrowser) {
        Start-Process $FrontendUrl
    }

    Write-Host ""
    Write-Host "RAVIN is ready for the demonstration."
}
catch {
    Write-Host ""
    Write-Host "=== RAVIN STARTUP FAILED ==="
    Write-Host $_.Exception.Message
    Write-Host ""
    Write-Host "Resolve the issue above and run Start-RAVIN again."

    exit 1
}
