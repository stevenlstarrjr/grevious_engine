#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec4 inColor;

layout(push_constant) uniform EditorOverlayConstants
{
    mat4 viewProjection;
} overlay;

layout(location = 0) out vec4 vertexColor;

void main()
{
    gl_Position = overlay.viewProjection * vec4(inPosition, 1.0);
    vertexColor = inColor;
}
