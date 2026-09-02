param(
    [string]$Executable = (Join-Path $PSScriptRoot '..\build\windows-port\bin\Release\genesis_render.exe'),
    [string]$OutputDirectory = (Join-Path $PSScriptRoot '..\captures\light_shaft_debug'),
    [int]$Width = 1280,
    [int]$Height = 720
)

$ErrorActionPreference = 'Stop'
$executablePath = (Resolve-Path -LiteralPath $Executable).Path
$outputPath = [System.IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Force -Path $outputPath | Out-Null

# This shot is intentionally unlike the broad atmosphere sweep: the camera
# faces the Sun and the selected broken-cloud offset puts the disk beside a
# real cloud opening so cloud bodies, additive shafts, and glare share a frame.
$common = @(
    '--validation', 'off',
    '--resolution', "$Width", "$Height",
    '--demo', 'atmosphere',
    '--clouds', 'on',
    '--cloud-coverage', '0.50',
    '--cloud-density', '1.0',
    '--cloud-offset', '90', '-24',
    '--time-of-day', 'on',
    '--day-cycle-seconds', '86400',
    '--time-hour', '18',
    '--atmosphere-camera-preset', '1',
    '--atmosphere-camera-pitch', '13',
    '--atmosphere-camera-yaw', '-110',
    '--visibility-mask', '0',
    '--taa', 'on',
    '--camera-input', 'off'
)

$captures = @(
    [pscustomobject]@{ Name = '01_native_beauty'; Label = 'Native beauty: HDR Sun + additive shafts'; Mode = 'final'; Path = 'rasterized' },
    [pscustomobject]@{ Name = '02_rr_beauty'; Label = 'Path + DLSS Ray Reconstruction beauty'; Mode = 'final'; Path = 'rr' },
    [pscustomobject]@{ Name = '03_cloud_opacity'; Label = 'Cloud opacity (white = opaque)'; Mode = 'transmittance'; Path = 'rasterized' },
    [pscustomobject]@{ Name = '04_shaft_illumination'; Label = 'Additive shaft illumination (white = maximum)'; Mode = 'shafts'; Path = 'rasterized' }
)

foreach ($capture in $captures) {
    $capturePath = Join-Path $outputPath ($capture.Name + '.png')
    if ($capture.Path -eq 'rr') {
        $arguments = $common + @(
            '--render-mode', 'path-traced', '--ray-tracing', 'required',
            '--path-max-bounces', '2', '--path-interactive-samples', '1',
            '--path-denoiser', 'ray-reconstruction',
            '--upscaler', 'dlss', '--upscaler-quality', 'quality',
            '--cloud-debug', $capture.Mode, '--exposure-ev', '-2.25',
            '--bloom', 'on', '--bloom-intensity', '0.10', '--bloom-radius', '1',
            '--frames', '64', '--capture-frame', '64', '--capture', $capturePath
        )
    } else {
        $beautyArguments = if ($capture.Mode -eq 'final') {
            @('--bloom', 'on', '--bloom-intensity', '0.10', '--bloom-radius', '1')
        } else {
            @('--bloom', 'off')
        }
        $exposure = if ($capture.Mode -eq 'final') { '-2.25' } else { '0' }
        $arguments = $common + $beautyArguments + @(
            '--render-mode', 'rasterized', '--ray-tracing', 'off',
            '--upscaler', 'native', '--cloud-debug', $capture.Mode,
            '--exposure-ev', $exposure,
            '--frames', '64', '--capture-frame', '64', '--capture', $capturePath
        )
    }
    Write-Host "Capturing $($capture.Label)"
    & $executablePath @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Genesis capture failed for $($capture.Name) with exit code $LASTEXITCODE"
    }
}

Add-Type -AssemblyName System.Drawing
$labelHeight = 44
$sheet = New-Object System.Drawing.Bitmap ($Width * 2), (($Height + $labelHeight) * 2)
$graphics = [System.Drawing.Graphics]::FromImage($sheet)
$graphics.Clear([System.Drawing.Color]::Black)
$graphics.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
$font = New-Object System.Drawing.Font 'Segoe UI', 15
for ($index = 0; $index -lt $captures.Count; ++$index) {
    $x = ($index % 2) * $Width
    $y = [Math]::Floor($index / 2) * ($Height + $labelHeight)
    $imagePath = Join-Path $outputPath ($captures[$index].Name + '.png')
    $source = [System.Drawing.Image]::FromFile($imagePath)
    try {
        $graphics.DrawImage($source, $x, $y, $Width, $Height)
    } finally {
        $source.Dispose()
    }
    $graphics.DrawString($captures[$index].Label, $font,
                         [System.Drawing.Brushes]::White, $x + 10, $y + $Height + 8)
}
$contactSheetPath = Join-Path $outputPath 'contact_sheet.png'
$sheet.Save($contactSheetPath, [System.Drawing.Imaging.ImageFormat]::Png)
$font.Dispose()
$graphics.Dispose()
$sheet.Dispose()
Write-Host "Contact sheet: $contactSheetPath"
