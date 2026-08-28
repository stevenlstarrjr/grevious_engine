#pragma once

#include "renderer/visibility/Visibility.hpp"

#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include <array>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace renderer::scene {

// Scene IR handles are compact, append-only indices. They remain stable for the
// lifetime of a Scene and are deliberately serializable/binding-friendly. The
// runtime residency system may later add generation-checked renderer handles.
template <typename Tag> class Handle final {
  public:
    static constexpr std::uint32_t invalidIndex = std::numeric_limits<std::uint32_t>::max();

    constexpr Handle() = default;
    [[nodiscard]] static constexpr Handle fromIndex(std::uint32_t index) { return Handle(index); }

    [[nodiscard]] constexpr bool valid() const { return index_ != invalidIndex; }
    [[nodiscard]] constexpr std::uint32_t index() const { return index_; }

    auto operator<=>(const Handle&) const = default;

  private:
    explicit constexpr Handle(std::uint32_t index) : index_(index) {}

    std::uint32_t index_ = invalidIndex;
};

using ImageHandle = Handle<struct ImageTag>;
using SamplerHandle = Handle<struct SamplerTag>;
using TextureHandle = Handle<struct TextureTag>;
using MaterialHandle = Handle<struct MaterialTag>;
using PrimitiveHandle = Handle<struct PrimitiveTag>;
using MeshHandle = Handle<struct MeshTag>;
using CameraHandle = Handle<struct CameraTag>;
using LightHandle = Handle<struct LightTag>;
using NodeHandle = Handle<struct NodeTag>;

static_assert(sizeof(NodeHandle) == sizeof(std::uint32_t));

enum class Filter : std::uint8_t { nearest, linear };
enum class MipFilter : std::uint8_t { none, nearest, linear };
enum class WrapMode : std::uint8_t { clampToEdge, mirroredRepeat, repeat };

struct Image {
    std::string name;
    std::string sourceUri;
    std::string mimeType;
    std::vector<std::byte> encodedData;
};

struct Sampler {
    std::string name;
    Filter minFilter = Filter::linear;
    Filter magFilter = Filter::linear;
    MipFilter mipFilter = MipFilter::linear;
    WrapMode wrapU = WrapMode::repeat;
    WrapMode wrapV = WrapMode::repeat;
};

struct Texture {
    std::string name;
    ImageHandle image;
    // An invalid handle selects the renderer's default sampler.
    SamplerHandle sampler;
};

struct TextureBinding {
    TextureHandle texture;
    std::uint32_t texCoord = 0;
};

// Surface scheduling is independent of visibility layers and render-graph
// passes. glTF alphaMode maps to the first three queues; additive is available
// to renderer-authored materials.
enum class MaterialQueue : std::uint8_t { opaque, alphaMasked, transparent, additive };

struct Material {
    std::string name;
    glm::vec4 baseColorFactor{1.0F};
    float metallicFactor = 1.0F;
    float roughnessFactor = 1.0F;
    glm::vec3 emissiveFactor{0.0F};
    TextureBinding baseColorTexture;
    TextureBinding metallicRoughnessTexture;
    TextureBinding normalTexture;
    float normalScale = 1.0F;
    TextureBinding occlusionTexture;
    float occlusionStrength = 1.0F;
    TextureBinding emissiveTexture;
    MaterialQueue queue = MaterialQueue::opaque;
    float alphaCutoff = 0.5F;
    bool doubleSided = false;
};

enum class PrimitiveTopology : std::uint8_t {
    points,
    lines,
    lineStrip,
    triangles,
    triangleStrip,
    triangleFan,
};

struct MeshPrimitive {
    std::string name;
    PrimitiveTopology topology = PrimitiveTopology::triangles;
    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec4> tangents;
    std::array<std::vector<glm::vec2>, 2> texCoords;
    std::vector<glm::vec4> colors;
    std::vector<std::uint32_t> indices;
    // Invalid selects the renderer's default material.
    MaterialHandle material;
};

struct Mesh {
    std::string name;
    std::vector<PrimitiveHandle> primitives;
};

enum class CameraType : std::uint8_t { perspective, orthographic };

struct Camera {
    std::string name;
    CameraType type = CameraType::perspective;
    float verticalFieldOfView = 0.785398163F;
    std::optional<float> aspectRatio;
    float zNear = 0.1F;
    // No far plane represents an infinite perspective projection.
    std::optional<float> zFar;
    float xMagnitude = 1.0F;
    float yMagnitude = 1.0F;
};

enum class LightType : std::uint8_t { directional, point, spot };

struct Light {
    std::string name;
    LightType type = LightType::point;
    glm::vec3 color{1.0F};
    float intensity = 1.0F;
    std::optional<float> range;
    float innerConeAngle = 0.0F;
    float outerConeAngle = 0.785398163F;
};

struct Node {
    std::string name;
    glm::mat4 localTransform{1.0F};
    MeshHandle mesh;
    CameraHandle camera;
    LightHandle light;
    RenderLayerMask visibilityMask = defaultInstanceRenderLayerMask;
    std::vector<NodeHandle> children;
};

struct AxisAlignedBounds {
    glm::vec3 minimum{0.0F};
    glm::vec3 maximum{0.0F};
    bool valid = false;
};

struct ValidationIssue {
    std::string path;
    std::string message;
};

struct ValidationReport {
    std::vector<ValidationIssue> issues;

    [[nodiscard]] bool ok() const { return issues.empty(); }
    [[nodiscard]] std::string describe() const;
};

class Scene final {
  public:
    explicit Scene(std::string name = {}, std::string sourceUri = {});

    [[nodiscard]] ImageHandle addImage(Image image);
    [[nodiscard]] SamplerHandle addSampler(Sampler sampler);
    [[nodiscard]] TextureHandle addTexture(Texture texture);
    [[nodiscard]] MaterialHandle addMaterial(Material material);
    [[nodiscard]] PrimitiveHandle addPrimitive(MeshPrimitive primitive);
    [[nodiscard]] MeshHandle addMesh(Mesh mesh);
    [[nodiscard]] CameraHandle addCamera(Camera camera);
    [[nodiscard]] LightHandle addLight(Light light);
    [[nodiscard]] NodeHandle addNode(Node node);

    void addRoot(NodeHandle node);
    void addChild(NodeHandle parent, NodeHandle child);

    [[nodiscard]] const std::string& name() const { return name_; }
    [[nodiscard]] const std::string& sourceUri() const { return sourceUri_; }
    [[nodiscard]] std::span<const Image> images() const { return images_; }
    [[nodiscard]] std::span<const Sampler> samplers() const { return samplers_; }
    [[nodiscard]] std::span<const Texture> textures() const { return textures_; }
    [[nodiscard]] std::span<const Material> materials() const { return materials_; }
    [[nodiscard]] std::span<const MeshPrimitive> primitives() const { return primitives_; }
    [[nodiscard]] std::span<const Mesh> meshes() const { return meshes_; }
    [[nodiscard]] std::span<const Camera> cameras() const { return cameras_; }
    [[nodiscard]] std::span<const Light> lights() const { return lights_; }
    [[nodiscard]] std::span<const Node> nodes() const { return nodes_; }
    [[nodiscard]] std::span<const NodeHandle> roots() const { return roots_; }

    [[nodiscard]] const Image& image(ImageHandle handle) const;
    [[nodiscard]] const Sampler& sampler(SamplerHandle handle) const;
    [[nodiscard]] const Texture& texture(TextureHandle handle) const;
    [[nodiscard]] const Material& material(MaterialHandle handle) const;
    [[nodiscard]] const MeshPrimitive& primitive(PrimitiveHandle handle) const;
    [[nodiscard]] const Mesh& mesh(MeshHandle handle) const;
    [[nodiscard]] const Camera& camera(CameraHandle handle) const;
    [[nodiscard]] const Light& light(LightHandle handle) const;
    [[nodiscard]] const Node& node(NodeHandle handle) const;

    [[nodiscard]] ValidationReport validate() const;

  private:
    std::string name_;
    std::string sourceUri_;
    std::vector<Image> images_;
    std::vector<Sampler> samplers_;
    std::vector<Texture> textures_;
    std::vector<Material> materials_;
    std::vector<MeshPrimitive> primitives_;
    std::vector<Mesh> meshes_;
    std::vector<Camera> cameras_;
    std::vector<Light> lights_;
    std::vector<Node> nodes_;
    std::vector<NodeHandle> roots_;
};

struct MeshInstance {
    NodeHandle node;
    MeshHandle mesh;
    glm::mat4 worldTransform{1.0F};
    RenderLayerMask visibilityMask = defaultInstanceRenderLayerMask;
};

struct SceneEvaluation {
    // Indexed by NodeHandle::index().
    std::vector<glm::mat4> nodeWorldTransforms;
    std::vector<MeshInstance> meshInstances;
};

[[nodiscard]] AxisAlignedBounds computeBounds(const MeshPrimitive& primitive);
[[nodiscard]] SceneEvaluation evaluate(const Scene& scene);

} // namespace renderer::scene
