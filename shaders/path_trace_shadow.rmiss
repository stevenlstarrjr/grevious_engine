#version 460
#extension GL_EXT_ray_tracing : require

layout(location = 1) rayPayloadInEXT float shadowVisibility;

void main()
{
    shadowVisibility = 1.0;
}
