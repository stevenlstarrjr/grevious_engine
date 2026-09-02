param(
    [string]$Executable = (Join-Path $PSScriptRoot '..\build\windows-port\bin\Release\genesis_render.exe'),
    [string]$OutputDirectory = (Join-Path $PSScriptRoot '..\captures\atmosphere_debug_sweep'),
    [int]$Width = 1280,
    [int]$Height = 720,
    [int]$CaptureFrame = 12,
    [double]$CloudOffsetX = 72.0,
    [double]$CloudOffsetY = -15.5,
    [ValidateRange(0.0, 1.0)]
    [double]$CloudCoverage = 0.42,
    [ValidateRange(0.1, 4.0)]
    [double]$CloudDensity = 1.0,
    [ValidateSet('final', 'transmittance', 'shafts')]
    [string]$CloudDebug = 'final'
)

$ErrorActionPreference = 'Stop'
$executablePath = (Resolve-Path -LiteralPath $Executable).Path
$outputPath = [System.IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Force -Path $outputPath | Out-Null

# The matrix deliberately separates one variable at a time: four Sun states,
# four horizontal headings, two upward views, a camera inside the cloud layer,
# and two orbital views. This makes visual regressions diagnosable instead of
# producing a collection of attractive but incomparable screenshots.
$views = @(
    [pscustomobject]@{ Name = '01_surface_horizon_morning'; Preset = 1; Pitch = 5; Yaw = 0; Hour = 8 },
    [pscustomobject]@{ Name = '02_surface_horizon_noon'; Preset = 1; Pitch = 5; Yaw = 0; Hour = 12 },
    [pscustomobject]@{ Name = '03_surface_horizon_sunset'; Preset = 1; Pitch = 5; Yaw = 180; Hour = 19 },
    [pscustomobject]@{ Name = '04_surface_horizon_night'; Preset = 1; Pitch = 5; Yaw = 0; Hour = 22 },
    [pscustomobject]@{ Name = '05_surface_mid_north'; Preset = 1; Pitch = 28; Yaw = 0; Hour = 12 },
    [pscustomobject]@{ Name = '06_surface_mid_east'; Preset = 1; Pitch = 28; Yaw = 90; Hour = 12 },
    [pscustomobject]@{ Name = '07_surface_mid_south_evening'; Preset = 1; Pitch = 28; Yaw = 180; Hour = 18 },
    [pscustomobject]@{ Name = '08_surface_mid_west'; Preset = 1; Pitch = 28; Yaw = 270; Hour = 12 },
    [pscustomobject]@{ Name = '09_surface_zenith_noon'; Preset = 1; Pitch = 78; Yaw = 0; Hour = 12 },
    [pscustomobject]@{ Name = '10_surface_zenith_sunset'; Preset = 1; Pitch = 78; Yaw = 180; Hour = 19 },
    [pscustomobject]@{ Name = '11_cloud_level_oblique'; Preset = 5; Pitch = 20; Yaw = 90; Hour = 12 },
    [pscustomobject]@{ Name = '12_orbit_day'; Preset = 9; Pitch = -85; Yaw = 0; Hour = 12 },
    [pscustomobject]@{ Name = '13_orbit_sunset'; Preset = 9; Pitch = -85; Yaw = 180; Hour = 19 }
)

$commonArguments = @(
    '--validation', 'off',
    '--resolution', "$Width", "$Height",
    '--render-mode', 'path-traced',
    '--ray-tracing', 'required',
    '--path-max-bounces', '2',
    '--path-interactive-samples', '1',
    '--path-denoiser', 'ray-reconstruction',
    '--upscaler', 'dlss',
    '--upscaler-quality', 'quality',
    '--demo', 'atmosphere',
    '--clouds', 'on',
    '--cloud-coverage', "$CloudCoverage",
    '--cloud-density', "$CloudDensity",
    '--cloud-offset', "$CloudOffsetX", "$CloudOffsetY",
    '--cloud-debug', $CloudDebug,
    '--time-of-day', 'on',
    '--day-cycle-seconds', '86400',
    '--visibility-mask', '0',
    '--taa', 'off',
    '--bloom', 'off',
    '--camera-input', 'off',
    '--frames', "$CaptureFrame",
    '--capture-frame', "$CaptureFrame"
)

$manifest = foreach ($view in $views) {
    $capturePath = Join-Path $outputPath ($view.Name + '.png')
    $arguments = $commonArguments + @(
        '--atmosphere-camera-preset', "$($view.Preset)",
        '--atmosphere-camera-pitch', "$($view.Pitch)",
        '--atmosphere-camera-yaw', "$($view.Yaw)",
        '--time-hour', "$($view.Hour)",
        '--capture', $capturePath
    )
    Write-Host "Capturing $($view.Name)"
    & $executablePath @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Genesis capture failed for $($view.Name) with exit code $LASTEXITCODE"
    }
    [pscustomobject]@{
        Name = $view.Name
        CameraPreset = $view.Preset
        PitchDegrees = $view.Pitch
        YawDegrees = $view.Yaw
        SolarHour = $view.Hour
        CloudCoverage = $CloudCoverage
        CloudDensity = $CloudDensity
        CloudDebug = $CloudDebug
        Image = $capturePath
    }
}

$manifestPath = Join-Path $outputPath 'manifest.csv'
$manifest | Export-Csv -LiteralPath $manifestPath -NoTypeInformation

try {
    Add-Type -AssemblyName System.Drawing
    $columns = 4
    $tileWidth = 480
    $imageHeight = 270
    $labelHeight = 30
    $tileHeight = $imageHeight + $labelHeight
    $rows = [int][Math]::Ceiling($views.Count / [double]$columns)
    $sheet = New-Object System.Drawing.Bitmap ($columns * $tileWidth), ($rows * $tileHeight)
    $graphics = [System.Drawing.Graphics]::FromImage($sheet)
    $graphics.Clear([System.Drawing.Color]::Black)
    $graphics.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
    $font = New-Object System.Drawing.Font 'Segoe UI', 11
    $brush = [System.Drawing.Brushes]::White
    for ($index = 0; $index -lt $views.Count; ++$index) {
        $column = $index % $columns
        $row = [int][Math]::Floor($index / [double]$columns)
        $x = $column * $tileWidth
        $y = $row * $tileHeight
        $imagePath = Join-Path $outputPath ($views[$index].Name + '.png')
        $source = [System.Drawing.Image]::FromFile($imagePath)
        try {
            $graphics.DrawImage($source, $x, $y, $tileWidth, $imageHeight)
        } finally {
            $source.Dispose()
        }
        $graphics.DrawString($views[$index].Name, $font, $brush, $x + 6, $y + $imageHeight + 5)
    }
    $contactSheetPath = Join-Path $outputPath 'contact_sheet.png'
    $sheet.Save($contactSheetPath, [System.Drawing.Imaging.ImageFormat]::Png)
    $font.Dispose()
    $graphics.Dispose()
    $sheet.Dispose()
    Write-Host "Contact sheet: $contactSheetPath"
} catch {
    Write-Warning "Captures completed, but the optional contact sheet failed: $_"
}

Write-Host "Manifest: $manifestPath"
