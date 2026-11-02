"""GPU regression for camera look controls and first-hit IDs. Requires NumPy and Pillow."""
import argparse
import csv
import json
import re
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image


def fixture(folder):
    blob = bytearray()
    views, accessors = [], []
    def attribute(values, kind, components, bounds=False):
        a = np.asarray(values, dtype="<f4" if kind == 5126 else "<u2")
        while len(blob) % 4:
            blob.append(0)
        views.append({"buffer":0,"byteOffset":len(blob),"byteLength":a.nbytes})
        blob.extend(a.tobytes())
        acc = {"bufferView":len(views)-1,"componentType":kind,
               "count":len(a),"type":components}
        if bounds:
            acc.update(min=a.min(axis=0).tolist(),max=a.max(axis=0).tolist())
        accessors.append(acc)
        return len(accessors)-1

    uv = attribute([(0,1),(1,1),(1,0),(0,0)],5126,"VEC2")
    normal = attribute([(0,0,1)]*4,5126,"VEC3")
    indices = attribute([0,1,2,0,2,3],5123,"SCALAR")
    def quad(lo,hi,material):
        pos=attribute([(-.5,lo,0),(.5,lo,0),(.5,hi,0),(-.5,hi,0)],5126,"VEC3",True)
        return {"attributes":{"POSITION":pos,"NORMAL":normal,"TEXCOORD_0":uv},
                "indices":indices,"material":material}
    meshes=[{"primitives":[quad(-.7,0,0),quad(0,.7,1)]},{"primitives":[quad(-.7,.7,2)]}]
    pixels=np.full((4,4,4),255,np.uint8)
    pixels[::2,::2,3]=0
    pixels[1::2,1::2,3]=0
    Image.fromarray(pixels).save(folder/"holes.png")
    doc={"asset":{"version":"2.0"},"scene":0,"scenes":[{"nodes":[0,1,2]}],
         "nodes":[{"mesh":0,"name":"Wagon left","translation":[-1.25,0,0]},
                  {"mesh":0,"name":"Wagon right","translation":[1.25,0,0]},
                  {"mesh":1,"name":"Cutout leaf"}],
         "meshes":meshes,"materials":[
             {"name":'Felt, "blue"',"pbrMetallicRoughness":{"baseColorFactor":[.08,.35,.8,1],"metallicFactor":.15,"roughnessFactor":.48}},
             {"name":"Emissive lantern","emissiveFactor":[.8,.2,.03],"pbrMetallicRoughness":{"baseColorFactor":[.1,.07,.04,1],"metallicFactor":0}},
             {"name":"Leaf","alphaMode":"MASK","alphaCutoff":.5,"pbrMetallicRoughness":{"baseColorTexture":{"index":0},"metallicFactor":0}}],
         "images":[{"uri":"holes.png"}],"textures":[{"source":0,"sampler":0}],
         "samplers":[{"magFilter":9728,"minFilter":9728}],"buffers":[{"byteLength":len(blob),"uri":"test.bin"}],
         "bufferViews":views,"accessors":accessors}
    (folder/"test.bin").write_bytes(blob)
    path=folder/"test.gltf"
    path.write_text(json.dumps(doc))
    return path


def pfm(path):
    with path.open("rb") as f:
        channels=3 if f.readline().strip()==b"PF" else 1
        w,h=map(int,f.readline().split())
        endian="<" if float(f.readline())<0 else ">"
        return np.frombuffer(f.read(),dtype=endian+"f4").reshape(h,w,channels)[::-1]


def rgb(path):
    return np.asarray(Image.open(path).convert("RGB"),dtype=np.int16)


def interior(mask):
    safe=np.zeros_like(mask)
    safe[2:-2,2:-2]=np.logical_and.reduce([mask[y:y+mask.shape[0]-4,x:x+mask.shape[1]-4]
                                              for y in range(5) for x in range(5)])
    return safe


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("renderer",type=Path)
    ap.add_argument("hdri",type=Path)
    ap.add_argument("output",type=Path)
    ns=ap.parse_args()
    exe,sky,out=ns.renderer.resolve(),ns.hdri.resolve(),ns.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    for invalid in (["--id-aovs"], ["--white-balance-k","1800"], ["--visible-sky-ev","nan"]):
        result = subprocess.run([str(exe), *invalid], capture_output=True, text=True)
        assert result.returncode != 0 and invalid[0] in result.stdout + result.stderr
    model=fixture(out)
    def render(name,*extra,backend="path-traced",frames=24,aovs=True,demo=False):
        folder=out/name
        folder.mkdir(exist_ok=True)
        args=[str(exe),"--render-mode",backend,"--environment",str(sky),"--frames",str(frames),
              "--resolution","720" if demo else "320","480" if demo else "200","--taa","off","--bloom","off",
              "--film-grain","0","--exposure-ev","1","--capture",str(folder/"capture.png"),"--capture-frame",str(frames),
              "--environment-rotation","-17.75","--validation","required"]
        args+=["--demo","hdri-backdrop"] if demo else ["--scene",str(model)]
        if backend=="path-traced":
            args+=["--ray-tracing","required"]
        if aovs:
            args+=["--aov-export",str(folder),"--aov-frame",str(frames)]
        with (folder/"renderer.log").open("w") as log:
            done=subprocess.run(args+list(extra),stdout=log,stderr=subprocess.STDOUT,check=False)
        log=(folder/"renderer.log").read_text(errors="replace")
        if done.returncode or re.search(r"\[ERROR\]|Validation Error|VUID-",log):
            raise RuntimeError(f"{name} failed; see {folder/'renderer.log'}\n{log[-1500:]}")
        print(name,flush=True)
        return folder/f"frame_{frames:08d}"

    base=render("baseline")
    ids=render("ids","--id-aovs")
    dim=render("sky","--visible-sky-ev","-6","--id-aovs")
    cool=render("white-balance","--white-balance-k","3200","--id-aovs")
    clay=render("clay","--clay","--id-aovs")
    again=render("id-repeat","--id-aovs",frames=8)
    obj=pfm(ids/"object_id.pfm")[:,:,0]
    mat=pfm(ids/"material_id.pfm")[:,:,0]
    assert set(np.unique(obj))=={0,1,2,3} and set(np.unique(mat))=={0,1,2,3}
    for i in (1,2):
        assert set(np.unique(mat[obj==i]))=={1,2}
    assert set(np.unique(mat[obj==3]))=={3}
    # The middle card has both solid texels and true background holes inside its bounds.
    yy,xx=np.where(obj==3)
    assert np.any(obj[yy.min()+1:yy.max(),xx.min()+1:xx.max()]==0)
    with (ids/"id_legend.csv").open(newline="",encoding="utf-8") as f:
        labels=list(csv.DictReader(f))
    assert {'Wagon left','Wagon right','Cutout leaf','Felt, "blue"','Emissive lantern','Leaf'} <= {r["name"] for r in labels}
    for passname,plane in (("object_id",obj),("material_id",mat)):
        colors=rgb(ids/(passname+".png"))
        for row in (r for r in labels if r["pass"]==passname):
            assert np.all(colors[plane==int(row["id"])]==[int(row[c]) for c in "rgb"])
        for other in (again,clay,cool,dim):
            assert np.array_equal(plane,pfm(other/(passname+".pfm"))[:,:,0])
    original=pfm(base/"raw_beauty.pfm")
    assert len(json.loads((base/"manifest.json").read_text())["planes"])==15
    assert len(json.loads((ids/"manifest.json").read_text())["planes"])==17
    assert np.array_equal(original,pfm(ids/"raw_beauty.pfm"))
    assert np.array_equal(original,pfm(cool/"raw_beauty.pfm"))
    shaded=pfm(dim/"raw_beauty.pfm")
    surface,background=interior(obj!=0),interior(obj==0)
    np.testing.assert_allclose(shaded[surface],original[surface],rtol=1e-5,atol=1e-6)
    np.testing.assert_allclose(shaded[background],original[background]/64,rtol=2e-5,atol=1e-6)
    albedo,rough=pfm(clay/"albedo.pfm"),pfm(clay/"roughness.pfm")
    np.testing.assert_allclose(albedo[surface],.18,atol=.0002)
    np.testing.assert_allclose(rough[surface],.85,atol=.001)
    assert pfm(clay/"direct_lighting.pfm")[interior(mat==2),0].mean()>.75
    baseline_rgb=rgb(base.parent/"capture.png")
    balanced=rgb(cool.parent/"capture.png")
    assert balanced[background,2].mean()>baseline_rgb[background,2].mean()
    assert balanced[background,0].mean()<baseline_rgb[background,0].mean()
    raster=render("raster",backend="rasterized",aovs=False)
    rasterdim=render("raster-sky","--visible-sky-ev","-6",backend="rasterized",aovs=False)
    rasterclay=render("raster-clay","--clay",backend="rasterized",aovs=False)
    rasterwb=render("raster-wb","--white-balance-k","3200",backend="rasterized",aovs=False)
    r=rgb(raster.parent/"capture.png")
    assert np.array_equal(r[surface],rgb(rasterdim.parent/"capture.png")[surface])
    assert r[background].mean()>rgb(rasterdim.parent/"capture.png")[background].mean()
    assert np.abs(r[surface]-rgb(rasterclay.parent/"capture.png")[surface]).mean()>2
    assert rgb(rasterwb.parent/"capture.png")[background,2].mean()>r[background,2].mean()
    print("Verified sky-only exposure, display-only white balance, clay/emission/cutouts, integer IDs, PNG colors and stable edges.",flush=True)
    if "blackridge" in sky.name.lower() or sky.name.startswith("BR_"):
        render("moon-default",demo=True,aovs=False)
        render("moon-visible-minus6","--visible-sky-ev","-6",demo=True,aovs=False)

if __name__=="__main__":
    main()
