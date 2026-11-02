param(
    [Parameter(Mandatory)][ValidateSet('wide','regional','zoom','stream-albedo','held-albedo','held-ref-albedo','ref-albedo','ref-beauty','small-cache')][string]$Trial,
    [Parameter(Mandatory)][string]$DataRoot
)
$ErrorActionPreference='Stop'
$proofRoot=(Resolve-Path -LiteralPath $DataRoot).Path
$engineDir=(Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$renderExe=Join-Path $engineDir 'build\windows\bin\Release\genesis_render.exe'
$assetsDir=Join-Path $proofRoot 'data'
$resultDir=Join-Path $proofRoot 'results'
$null=New-Item -ItemType Directory -Force -Path $resultDir
$useStream=$Trial -notin 'ref-beauty','ref-albedo','held-ref-albedo'
$globe=if($useStream){'earth-streamed.gltf'}else{'earth-reference.gltf'}
$frames=switch($Trial){'wide'{80};'regional'{61};'held-albedo'{160};'held-ref-albedo'{160};default{176}}
$argsForRender=@('--scene',(Join-Path $assetsDir $globe),'--environment',(Join-Path $assetsDir 'studio.hdr'),
    '--render-mode','path-traced','--ray-tracing','required','--resolution','960','640',
    '--frames',"$frames",'--capture-frame',"$frames",'--capture',(Join-Path $resultDir "$Trial.png"),
    '--validation','required','--path-interactive-samples','1','--taa','off','--bloom','off')
if($useStream){
    $pool=if($Trial -eq 'small-cache'){2}else{16}
    $argsForRender+=@('--virtual-base-color',(Join-Path $assetsDir 'earth.grvt'),'--vt-pool-mib',"$pool")
}
if($Trial -ne 'wide'){$argsForRender+=@('--camera-script','zoom-hold')}
if($Trial -like '*albedo'){$argsForRender+=@('--path-debug','albedo')}
$logFile=Join-Path $resultDir "$Trial.log"
Push-Location -LiteralPath $engineDir
try {
    & $renderExe @argsForRender *> $logFile
    $exitCode=$LASTEXITCODE
    if($exitCode -ne 0){Get-Content -LiteralPath $logFile -Tail 15;throw "Renderer exit $exitCode"}
    $errorsFound=@(Select-String -LiteralPath $logFile -Pattern 'Validation Error|\[ERROR\]|VUID-')
    if($errorsFound.Count){$errorsFound|Select-Object -First 6|ForEach-Object {$_.Line};throw 'Renderer validation errors'}
    Select-String -LiteralPath $logFile -Pattern 'VT_CONFIG|VT_FRAME frame=(64|160) |Captured frame|Clean shutdown' |ForEach-Object {$_.Line}
}finally{Pop-Location}
