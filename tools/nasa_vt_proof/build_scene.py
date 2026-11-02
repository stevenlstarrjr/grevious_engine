"""Make the matched streamed and conventional NASA test scenes."""
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

OUT = Path(sys.argv[1]).resolve()/"data"
W,H = 65536,32768
X,Y,S = 34176,8320,512
lon0 = (X+260)/W*2*math.pi-math.pi
lat0 = math.pi/2-(Y+336)/H*math.pi
xs,ys = 256,128
vertex=[]
uv=[]
tri=[]
for y in range(ys+1):
    lat=math.pi/2-y/ys*math.pi
    for x in range(xs+1):
        lon=x/xs*2*math.pi-math.pi-lon0
        a,b,c=math.cos(lat)*math.sin(lon),math.sin(lat),math.cos(lat)*math.cos(lon)
        vertex.append((a,b*math.cos(lat0)-c*math.sin(lat0), b*math.sin(lat0)+c*math.cos(lat0)))
        uv.append((x/xs,y/ys))
        if x<xs and y<ys:
            i=y*(xs+1)+x
            if y>0: tri.extend((i,i+xs+1,i+1))
            if y<ys-1: tri.extend((i+1,i+xs+1,i+xs+2))
vertex=np.array(vertex,'<f4')
uv=np.array(uv,'<f4')
idx=np.array(tri,'<u4')
Image.new('RGB',(2,2),'white').save(OUT/'placeholder.png')

def export(name, image, coords):
    pieces=(vertex.tobytes(),coords.astype('<f4').tobytes(),idx.tobytes())
    offsets=[0,len(pieces[0]),len(pieces[0])+len(pieces[1])]
    blob=b''.join(pieces)
    (OUT/(name+'.bin')).write_bytes(blob)
    doc={
      'asset':{'version':'2.0','generator':'Grevious NASA VT proof'},
      'scene':0,'scenes':[{'nodes':[0]}],'nodes':[{'mesh':0}],
      'meshes':[{'primitives':[{'attributes':{'POSITION':0,'NORMAL':0,'TEXCOORD_0':1},'indices':2,'material':0}]}],
      'buffers':[{'uri':name+'.bin','byteLength':len(blob)}],
      'bufferViews':[{'buffer':0,'byteOffset':offset,'byteLength':len(piece)} for offset,piece in zip(offsets,pieces)],
      'accessors':[
          {'bufferView':0,'componentType':5126,'count':len(vertex),'type':'VEC3','min':vertex.min(axis=0).tolist(),'max':vertex.max(axis=0).tolist()},
          {'bufferView':1,'componentType':5126,'count':len(vertex),'type':'VEC2'},
          {'bufferView':2,'componentType':5125,'count':len(idx),'type':'SCALAR'}],
      'materials':[{'name':'NASA July Earth','pbrMetallicRoughness':{'baseColorTexture':{'index':0},'metallicFactor':0,'roughnessFactor':1}}],
      'images':[{'uri':image}],'textures':[{'sampler':0,'source':0}],
      'samplers':[{'magFilter':9729,'minFilter':9987,'wrapS':10497,'wrapT':33071}]
    }
    (OUT/(name+'.gltf')).write_text(json.dumps(doc))

export('earth-streamed','placeholder.png',uv)
export('earth-reference','corsica-reference.png',(uv*[W,H]-[X,Y])/S)
# An actual lat-long lighting map. A broad softbox sits above and left of camera.
w,h=64,32
yy,xx=np.mgrid[0:h,0:w]
theta=(yy+.5)*math.pi/h
phi=((xx+.5)/w-.5)*2*math.pi
direction=np.stack([np.sin(theta)*np.cos(phi),np.cos(theta),np.sin(theta)*np.sin(phi)],axis=-1)
key=np.array([-.42,.42,.8]);key/=np.linalg.norm(key)
v=.012+9*np.maximum(0.,direction@key)**9
mantissa,exponent=np.frexp(v)
pixels=np.zeros((h,w,4),np.uint8)
pixels[:,:,:3]=(mantissa*256).astype(np.uint8)[:,:,None]
pixels[:,:,3]=exponent+128
(OUT/'studio.hdr').write_bytes(b'#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 32 +X 64\n'+pixels.tobytes())
