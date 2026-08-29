param(
    [string]$Destination = (Join-Path $PSScriptRoot "..\thirdparty\Streamline")
)

$ErrorActionPreference = "Stop"
$releaseUrl = "https://github.com/NVIDIA-RTX/Streamline/releases/download/v2.12.0/streamline-sdk-v2.12.0.zip"
$expectedSha256 = "F5C0A3D870707DDDC3570FB4BCD3655CF48A8A68C3A9D342910CFA21B77DCF48"
$archive = Join-Path ([System.IO.Path]::GetTempPath()) "streamline-sdk-v2.12.0.zip"
$extractRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("genesis-streamline-" + [guid]::NewGuid())
$resolvedDestination = [System.IO.Path]::GetFullPath($Destination)

try {
    Write-Host "Downloading NVIDIA Streamline SDK 2.12.0..."
    Invoke-WebRequest -Uri $releaseUrl -OutFile $archive
    $actualSha256 = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash
    if ($actualSha256 -ne $expectedSha256) {
        throw "Streamline archive SHA-256 mismatch. Expected $expectedSha256, got $actualSha256"
    }

    Expand-Archive -LiteralPath $archive -DestinationPath $extractRoot -Force
    $sdkRoot = Get-ChildItem -LiteralPath $extractRoot -Directory | Select-Object -First 1
    if ($null -eq $sdkRoot) {
        throw "The Streamline archive did not contain an SDK directory"
    }

    New-Item -ItemType Directory -Force -Path $resolvedDestination | Out-Null
    Copy-Item -Recurse -Force -LiteralPath (Join-Path $sdkRoot.FullName "include") -Destination $resolvedDestination
    New-Item -ItemType Directory -Force -Path (Join-Path $resolvedDestination "bin\x64") | Out-Null
    New-Item -ItemType Directory -Force -Path (Join-Path $resolvedDestination "docs") | Out-Null

    $runtimeFiles = @(
        "sl.interposer.dll", "sl.common.dll", "sl.dlss.dll", "sl.dlss_d.dll",
        "nvngx_dlss.dll", "nvngx_dlssd.dll", "nvngx_dlss.license.txt"
    )
    foreach ($file in $runtimeFiles) {
        Copy-Item -Force -LiteralPath (Join-Path $sdkRoot.FullName "bin\x64\$file") -Destination (Join-Path $resolvedDestination "bin\x64")
    }

    $docFiles = @(
        "ProgrammingGuide.md", "ProgrammingGuideManualHooking.md",
        "ProgrammingGuideDLSS.md", "ProgrammingGuideDLSS_RR.md"
    )
    foreach ($file in $docFiles) {
        Copy-Item -Force -LiteralPath (Join-Path $sdkRoot.FullName "docs\$file") -Destination (Join-Path $resolvedDestination "docs")
    }
    Write-Host "Streamline SDK 2.12.0 installed at $resolvedDestination"
}
finally {
    if (Test-Path -LiteralPath $extractRoot) {
        Remove-Item -Recurse -Force -LiteralPath $extractRoot
    }
}
