# NASA Blue Marble texture streaming proof

This optional Windows Vulkan path replaces opaque glTF base-color textures with one
disk-backed color map in the Grevious closest-hit shader. The normal renderer uses
its existing material textures when the option is absent. The test follows a sphere
from orbit to Corsica, then holds the camera for a comparison in the same renderer.

## What the test measures

The July 2004 Earth map comes from NASA Earth Observatory. Eight 21,600-pixel-square
source quadrants make an 86,400 × 43,200 image. The packer resamples each quadrant
to 16,384 pixels on a side and assembles a 65,536 × 32,768 virtual texture.
The full map has 2,147,483,648 base-level texels, 17 mip levels and 174,770
individually compressed 128 × 128 pages. The packer processes one quadrant at a
time and saves a separate source crop before packing its pixels.
[NASA Blue Marble Next Generation, topography and bathymetry](https://science.nasa.gov/earth/earth-observatory/blue-marble-next-generation/base-topography-bathymetry/)

On the RTX 2060 SUPER, the 16 MiB pool loaded 795 pages and replaced 539 during
the 176-frame camera move and hold. It read 24,152,031 compressed payload bytes,
in addition to the pack header and index. All eight pages requested on the held
view were resident, including native-resolution detail. Allocated buffer sizes were 16 MiB for texels,
721,028 bytes for the page table and feedback, and up to 4 MiB for transfer staging.
The engine's other resources and allocator overhead are outside this accounting.

The reference run uses a conventional, fully resident 512 × 512 crop on the
same sphere with remapped UVs. All visible close-up texels lie inside the crop.
The raw RGB bytes of the crop match the corresponding packed pages exactly.
The rendered 960 × 640 base-color comparison matched within one 8-bit level per
channel on 99.9987% of pixels. The verifier also checks the lit image, another
frame during the hold and eviction with a 2 MiB pool.

## Reproduce

Use a Release build of this branch, Developer PowerShell, and Python 3.11+
with Pillow and NumPy. Set a data directory on an SSD and run these commands
from the repository root. The location needs approximately 4 GB for the
source images, packed texture and captures in addition to the engine build.

```powershell
$dataRoot = Join-Path $env:LOCALAPPDATA 'Grevious-NASA-VT'
$toolsDir = '.\tools\nasa_vt_proof'
& "$toolsDir\download.ps1" -DataRoot $dataRoot
python "$toolsDir\build_map.py" $dataRoot
python "$toolsDir\build_scene.py" $dataRoot
foreach ($trial in 'wide','regional','zoom','ref-beauty','stream-albedo','ref-albedo','small-cache','held-albedo','held-ref-albedo') {
    & "$toolsDir\run_trial.ps1" -DataRoot $dataRoot -Trial $trial
}
python "$toolsDir\verify.py" $dataRoot
```

The logs, PNGs and `verification.json` appear under `results`. The packer records
the source JPEG hashes, dimensions and pack size in `data/source.json`.

## Implementation boundaries

The coarsest page stays resident. The path tracer computes a texture footprint at
ray hits and requests missing pages at the two mip levels needed for trilinear
filtering. Resident-page use is sampled for eviction. A miss uses a coarser
resident page until the next upload. Changed mappings reset accumulation and
denoising history. The streamed file is read and decompressed on the CPU, then
uploaded through Vulkan staging into a device-local buffer. The proof synchronizes
feedback once per frame and uploads at most 64 pages per batch.

This implements one sRGB color pack with horizontal wrap and vertical clamp.
It does not stream geometry, alpha masks, normal maps or displacement, and uses
neither DirectStorage nor GPU decompression. The reference comparison certifies
the held close-up. Frames during camera motion may show a coarse fallback.
