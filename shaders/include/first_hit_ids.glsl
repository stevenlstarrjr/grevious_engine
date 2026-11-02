#ifndef GENESIS_FIRST_HIT_IDS_GLSL
#define GENESIS_FIRST_HIT_IDS_GLSL
const uint ID_ONLY_RAY_DEPTH = 0xffffffffu;
layout(buffer_reference, std430, buffer_reference_align = 8) writeonly buffer FirstHitIds {
    uvec2 pixels[];
};

// The address is nonzero only on the requested AOV export frame. IDs stay
// integers throughout capture and bypass filtering, reconstruction and grading.
void writeFirstHitIds(uvec2 ids)
{
    uint pixel = gl_LaunchIDEXT.y * gl_LaunchSizeEXT.x + gl_LaunchIDEXT.x;
    FirstHitIds target = FirstHitIds(frameData.lookOptions.zw);
    target.pixels[pixel] = ids;
}
#endif
