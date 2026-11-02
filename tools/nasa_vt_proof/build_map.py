"""Pack NASA Blue Marble's eight July source quadrants, one quadrant at a time."""
import hashlib
import json
import math
import struct
import sys
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(sys.argv[1]).resolve()
OUT = ROOT / "data"
OUT.mkdir(exist_ok=True)
EDGE = 128
WIDTH, HEIGHT = 65536, 32768
LEVELS = WIDTH.bit_length()
Image.MAX_IMAGE_PIXELS = 21600 ** 2

levels = [(max(1, WIDTH >> i), max(1, HEIGHT >> i)) for i in range(LEVELS)]
starts = []
count = 0
for w, h in levels:
    starts.append(count)
    count += math.ceil(w / EDGE) * math.ceil(h / EDGE)
index = [(0, 0)] * count
header = struct.pack("<8s4I", b"GRVT001\0", WIDTH, HEIGHT, EDGE, LEVELS)

def compress_page(page):
    return zlib.compress(page.tobytes(), 1)

def emit_image(file, pool, image, level, xbase, ybase):
    w, h = image.size
    virtual_w = levels[level][0]
    across = math.ceil(virtual_w / EDGE)
    for y in range(0, h, EDGE):
        strip = np.asarray(image.crop((0, y, w, min(y+EDGE,h))).convert("RGBA"))
        if strip.shape[0] < EDGE or w % EDGE:
            strip = np.pad(strip, ((0,EDGE-strip.shape[0]), (0,(-w)%EDGE), (0,0)), mode="edge")
        pages = [strip[:, x:x+EDGE] for x in range(0, w, EDGE)]
        first = starts[level] + (ybase//EDGE+y//EDGE)*across + xbase//EDGE
        for page_id, compressed in enumerate(pool.map(compress_page, pages),first):
            assert index[page_id] == (0, 0)
            index[page_id] = (file.tell(),len(compressed))
            file.write(compressed)

def build():
    sources = []
    tail = Image.new("RGB",(512,256))
    target = OUT/"earth.grvt"
    partial = target.with_suffix(".part")
    with partial.open("w+b") as f, ThreadPoolExecutor(6) as pool:
        f.write(header)
        f.seek(len(header)+12*count)
        for row in range(2):
            for col in range(4):
                key = chr(65+col) + str(row+1)
                source = ROOT/"source"/f"world.topo.bathy.200407.3x21600x21600.{key}.jpg"
                with source.open("rb") as raw:
                    digest = hashlib.file_digest(raw,"sha256").hexdigest()
                with Image.open(source) as original:
                    assert original.size == (21600,21600)
                    image = original.resize((16384,16384),Image.Resampling.LANCZOS)
                sources.append(dict(tile=key,sha256=digest,bytes=source.stat().st_size))
                if key == "C1":
                    # Independent fully resident reference for the native-detail view.
                    x, y = 34176-32768, 8320
                    image.crop((x,y,x+512,y+512)).save(OUT/"corsica-reference.png")
                for level in range(7):
                    n = 16384 >> level
                    emit_image(f,pool,image,level,col*n,row*n)
                    image = image.resize((n//2,n//2),Image.Resampling.BOX)
                tail.paste(image,(128*col,128*row))
                print("Packed",key,f.tell(),flush=True)
        for level in range(7, LEVELS):
            emit_image(f,pool,tail,level,0,0)
            if level+1 < LEVELS:
                tail = tail.resize(levels[level+1],Image.Resampling.BOX)
        assert all(n for _,n in index)
        f.seek(len(header))
        for offset,size in index:
            f.write(struct.pack("<QI",offset,size))
    partial.replace(target)
    (OUT/"source.json").write_text(json.dumps(dict(
        source="NASA Earth Observatory, Blue Marble Next Generation, July 2004, topography and bathymetry",
        url="https://science.nasa.gov/earth/earth-observatory/blue-marble-next-generation/base-topography-bathymetry/",
        original=[86400,43200],virtual=[WIDTH,HEIGHT],page_edge=EDGE,levels=LEVELS,
        pages=count,disk_bytes=target.stat().st_size,quadrants=sources,
        reference_crop=[34176,8320,512,512]),indent=2))
    print("Finished",target.stat().st_size,flush=True)

if __name__ == "__main__":
    build()
