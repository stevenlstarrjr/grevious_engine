"""Verify source bytes, matched captures, budget, eviction and settling."""
import json
import re
import struct
import sys
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

root = Path(sys.argv[1]).resolve()
data = root / "data"
results = root / "results"
source = json.loads((data / "source.json").read_text())
report = {"source": source["source"], "virtual": source["virtual"], "trials": {}}

# Compare the packed level-zero pages against the reference PNG saved directly
# from the source quadrant before its bytes went through the tile writer.
x, y, w, h = source["reference_crop"]
edge = source["page_edge"]
reconstructed = np.empty((h, w, 3), np.uint8)
with (data/"earth.grvt").open("rb") as f:
    for oy in range(0, h, edge):
        for ox in range(0, w, edge):
            page = ((y+oy)//edge)*(source["virtual"][0]//edge)+(x+ox)//edge
            f.seek(24+page*12)
            offset, length = struct.unpack("<QI", f.read(12))
            f.seek(offset)
            rgba = np.frombuffer(zlib.decompress(f.read(length)), np.uint8).reshape(edge,edge,4)
            reconstructed[oy:oy+edge,ox:ox+edge] = rgba[:,:,:3]
assert np.array_equal(reconstructed, np.asarray(Image.open(data/"corsica-reference.png")))
report["source_crop_exact"] = True

def fields(line):
    return {k:int(v) for k,v in re.findall(r"(\w+)=(-?\d+)\b",line)}

for name in ("wide", "zoom", "small-cache", "ref-beauty", "stream-albedo", "ref-albedo"):
    log = (results/(name+".log")).read_text(encoding="utf-8-sig")
    assert "Clean shutdown" in log and not re.search(r"\[ERROR\]|VUID-|Validation Error",log)
    assert "NVIDIA GeForce RTX 2060 SUPER" in log and "validation enabled" in log
    gpu = float(re.search(r"GPU timing: path tracing ([\d.]+) ms",log).group(1))
    trial = {"gpu_trace_ms_over_route":gpu}
    if "VT_CONFIG" in log:
        cfg = fields(re.search(r"VT_CONFIG[^\n]+", log).group())
        frames = [fields(line) for line in re.findall(r"VT_FRAME[^\n]+",log)]
        tail = frames[-1]
        upper = sum(cfg[k] for k in ("poolBytes","tableFeedbackBytes","maxStagingBytes"))
        assert upper < 256 * 2**20
        assert tail["missing"] == 0 and tail["loaded"] == 0
        assert tail["diskPayloadBytes"] < cfg["fileBytes"] and tail["totalLoads"] < cfg["pages"]
        assert tail["resident"] * edge**2 * 4 <= cfg["poolBytes"]
        if name != "wide":
            assert tail["finestMip"] == 0 and tail["evictions"] > 0
        trial.update(config=cfg, final_counters=tail, max_gpu_stream_resources_bytes=upper,
                     max_missing=max(f["missing"] for f in frames))
    report["trials"][name] = trial

def compare(a, b):
    def rgb(name):
        return np.asarray(Image.open(results/(name+".png")).convert("RGB"),dtype=np.int16)
    diff = np.abs(rgb(a)-rgb(b))
    return dict(mean_absolute_channel_error=float(diff.mean()), max_channel_error=int(diff.max()),
                percent_pixels_all_channels_within_one=float(np.mean(diff.max(axis=2)<=1)*100))

report["albedo_difference_8bit"] = compare("stream-albedo","ref-albedo")
report["beauty_difference_8bit"] = compare("zoom","ref-beauty")
report["small_cache_beauty_difference_8bit"] = compare("small-cache","ref-beauty")
report["earlier_hold_albedo_difference_8bit"] = compare("held-albedo","held-ref-albedo")
assert report["albedo_difference_8bit"]["percent_pixels_all_channels_within_one"] > 99.9
assert report["earlier_hold_albedo_difference_8bit"]["percent_pixels_all_channels_within_one"] > 99.9
assert report["beauty_difference_8bit"]["mean_absolute_channel_error"] < .5
(results/"verification.json").write_text(json.dumps(report,indent=2))
print(json.dumps({k:report[k] for k in report if "difference" in k or k=="source_crop_exact"},indent=2))
