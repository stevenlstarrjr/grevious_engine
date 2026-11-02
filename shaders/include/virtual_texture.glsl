#ifndef GENESIS_VIRTUAL_TEXTURE_GLSL
#define GENESIS_VIRTUAL_TEXTURE_GLSL
layout(buffer_reference, std430, buffer_reference_align=4) buffer VirtualState { uint words[]; };
layout(buffer_reference, std430, buffer_reference_align=4) readonly buffer VirtualPixels { uint words[]; };

vec4 virtualTexel(VirtualState vt, ivec2 pos, uint level) {
    const int edge = 128;
    ivec2 extent=max(ivec2(1),ivec2(vt.words[0],vt.words[1])>>level);
    pos=ivec2((pos.x%extent.x+extent.x)%extent.x,clamp(pos.y,0,extent.y-1));
    for(uint mip=level; mip<vt.words[2]; ++mip) {
        uint page=vt.words[8u+mip]+uint(pos.y/edge)*uint((extent.x+edge-1)/edge)+uint(pos.x/edge);
        uint slot=vt.words[vt.words[3]+page];
        // Every miss requests its page. Resident hits only refresh LRU; sample
        // those on an 8x8 lattice to avoid millions of identical PCIe atomics.
        uvec2 probe=gl_LaunchIDEXT.xy+uvec2(controls.sampleIndex,controls.sampleIndex>>3);
        if(mip==level && (slot==0u || all(equal(probe&uvec2(7u),uvec2(0u))))) {
            atomicOr(vt.words[vt.words[4]+(page>>5)],1u<<(page&31u));
        }
        if(slot!=0u) {
            uint texel=(slot-1u)*16384u+uint(pos.y%edge)*128u+uint(pos.x%edge);
            vec4 v=unpackUnorm4x8(VirtualPixels(controls.virtualPixels).words[texel]);
            v.rgb=mix(pow((v.rgb+0.055)/1.055,vec3(2.4)),v.rgb/12.92,lessThanEqual(v.rgb,vec3(0.04045)));
            return v;
        }
        pos=pos>>1;
        extent=max(ivec2(1),extent>>1);
    }
    return vec4(0.0,0.0,0.0,1.0);
}

vec4 virtualLinear(VirtualState vt, vec2 uv, uint mip) {
    vec2 xy=uv*vec2(max(uvec2(1),uvec2(vt.words[0],vt.words[1])>>mip))-0.5;
    ivec2 ij=ivec2(floor(xy));
    vec2 f=fract(xy);
    return mix(mix(virtualTexel(vt,ij,mip),virtualTexel(vt,ij+ivec2(1,0),mip),f.x),
               mix(virtualTexel(vt,ij+ivec2(0,1),mip),virtualTexel(vt,ij+ivec2(1,1),mip),f.x),f.y);
}

vec4 sampleVirtual(VirtualState vt, vec2 uv, float desiredLod) {
    float lod=clamp(desiredLod,0.0,float(vt.words[2]-1u));
    uint mip=uint(floor(lod));
    vec4 a=virtualLinear(vt,uv,mip);
    if(fract(lod)<1e-5) return a;
    return mix(a,virtualLinear(vt,uv,mip+1u),fract(lod));
}
#endif
