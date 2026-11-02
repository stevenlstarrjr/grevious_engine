param([Parameter(Mandatory)][string]$DataRoot)
$ErrorActionPreference='Stop'
$dest=New-Item -ItemType Directory -Force -Path (Join-Path $DataRoot 'source')
$base='https://assets.science.nasa.gov/content/dam/science/esd/eo/images/bmng/bmng-topography-bathymetry/july'
foreach($tile in 'A1','B1','C1','D1','A2','B2','C2','D2') {
    $name="world.topo.bathy.200407.3x21600x21600.$tile.jpg"
    $file=Join-Path $dest.FullName $name
    if(Test-Path -LiteralPath $file){continue}
    curl.exe --fail --location --silent --show-error --retry 3 -o "$file.part" "$base/$name"
    if($LASTEXITCODE -ne 0){throw "NASA download failed for $tile"}
    Move-Item -LiteralPath "$file.part" -Destination $file
}
