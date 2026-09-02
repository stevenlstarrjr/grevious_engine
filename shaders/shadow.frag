#version 460
#extension GL_EXT_nonuniform_qualifier : require

struct MaterialRecord
{
    vec4 baseColorFactor;
    vec4 emissiveAlphaCutoff;
    vec4 metallicRoughnessOcclusionNormal;
    uvec4 textureIndices;
    uvec4 textureState;
};

layout(std430, set = 0, binding = 8) readonly buffer Materials
{
    MaterialRecord materials[];
};

layout(set = 1, binding = 0) uniform sampler2D materialTextures[];

layout(location = 0) in vec2 uv;
layout(location = 1) in vec2 uv1;
layout(location = 2) in flat uint materialIndex;
layout(location = 3) in vec4 vertexColor;

const uint BASE_COLOR_TEXTURE_BIT = 1u << 0u;
const uint ALPHA_MASK_MATERIAL_BIT = 1u << 0u;

void main()
{
    MaterialRecord material = materials[materialIndex];
    if ((material.textureState.z & ALPHA_MASK_MATERIAL_BIT) == 0u) {
        return;
    }

    float alpha = material.baseColorFactor.a * vertexColor.a;
    if ((material.textureState.y & BASE_COLOR_TEXTURE_BIT) != 0u) {
        vec2 baseColorUv =
            (material.textureState.w & BASE_COLOR_TEXTURE_BIT) != 0u ? uv1 : uv;
        alpha *= texture(materialTextures[nonuniformEXT(material.textureIndices.x)],
                         baseColorUv).a;
    }
    if (alpha < material.emissiveAlphaCutoff.a) {
        discard;
    }
}
