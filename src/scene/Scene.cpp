#include "renderer/scene/Scene.hpp"

#include <algorithm>
#include <cmath>
#include <deque>
#include <numbers>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <unordered_set>
#include <utility>

namespace renderer::scene {
namespace {

template <typename HandleType, typename Value>
HandleType append(std::vector<Value>& values, Value value, std::string_view resourceName) {
    if (values.size() >= HandleType::invalidIndex) {
        throw std::length_error(std::string(resourceName) + " handle space is exhausted");
    }
    const auto handle = HandleType::fromIndex(static_cast<std::uint32_t>(values.size()));
    values.push_back(std::move(value));
    return handle;
}

template <typename Value, typename HandleType>
const Value& lookup(const std::vector<Value>& values, HandleType handle,
                    std::string_view resourceName) {
    if (!handle.valid() || static_cast<std::size_t>(handle.index()) >= values.size()) {
        throw std::out_of_range(std::string("Invalid ") + std::string(resourceName) + " handle");
    }
    return values[handle.index()];
}

template <typename HandleType> bool resolves(HandleType handle, std::size_t count) {
    return handle.valid() && static_cast<std::size_t>(handle.index()) < count;
}

std::string elementPath(std::string_view collection, std::size_t index) {
    return std::string(collection) + '[' + std::to_string(index) + ']';
}

void issue(ValidationReport& report, std::string path, std::string message) {
    report.issues.push_back({.path = std::move(path), .message = std::move(message)});
}

bool finite(float value) { return std::isfinite(value); }

bool finite(const glm::vec2& value) { return finite(value.x) && finite(value.y); }

bool finite(const glm::vec3& value) {
    return finite(value.x) && finite(value.y) && finite(value.z);
}

bool finite(const glm::vec4& value) {
    return finite(value.x) && finite(value.y) && finite(value.z) && finite(value.w);
}

bool finite(const glm::mat4& value) {
    for (glm::mat4::length_type column = 0; column < 4; ++column) {
        if (!finite(value[column])) {
            return false;
        }
    }
    return true;
}

bool inRange(float value, float minimum, float maximum) {
    return finite(value) && value >= minimum && value <= maximum;
}

template <typename Vector>
void validateFiniteElements(ValidationReport& report, const Vector& values,
                            const std::string& path) {
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (!finite(values[index])) {
            issue(report, path + '[' + std::to_string(index) + ']',
                  "contains a non-finite component");
            return;
        }
    }
}

void validateBinding(ValidationReport& report, const TextureBinding& binding,
                     std::size_t textureCount, const std::string& path) {
    if (!binding.texture.valid()) {
        return;
    }
    if (!resolves(binding.texture, textureCount)) {
        issue(report, path + ".texture", "references a texture outside this scene");
    }
    if (binding.texCoord >= 2) {
        issue(report, path + ".texCoord", "only texture-coordinate sets 0 and 1 are supported");
    }
}

bool valid(Filter value) {
    switch (value) {
    case Filter::nearest:
    case Filter::linear:
        return true;
    }
    return false;
}

bool valid(MipFilter value) {
    switch (value) {
    case MipFilter::none:
    case MipFilter::nearest:
    case MipFilter::linear:
        return true;
    }
    return false;
}

bool valid(WrapMode value) {
    switch (value) {
    case WrapMode::clampToEdge:
    case WrapMode::mirroredRepeat:
    case WrapMode::repeat:
        return true;
    }
    return false;
}

bool valid(MaterialQueue value) {
    switch (value) {
    case MaterialQueue::opaque:
    case MaterialQueue::alphaMasked:
    case MaterialQueue::transparent:
    case MaterialQueue::additive:
        return true;
    }
    return false;
}

std::optional<std::string_view> topologyError(PrimitiveTopology topology,
                                              std::size_t elementCount) {
    switch (topology) {
    case PrimitiveTopology::points:
        return elementCount >= 1 ? std::nullopt
                                 : std::optional<std::string_view>("requires at least one vertex");
    case PrimitiveTopology::lines:
        return elementCount >= 2 && elementCount % 2 == 0
                   ? std::nullopt
                   : std::optional<std::string_view>(
                         "requires a positive vertex/index count divisible by two");
    case PrimitiveTopology::lineStrip:
        return elementCount >= 2
                   ? std::nullopt
                   : std::optional<std::string_view>("requires at least two vertices/indices");
    case PrimitiveTopology::triangles:
        return elementCount >= 3 && elementCount % 3 == 0
                   ? std::nullopt
                   : std::optional<std::string_view>(
                         "requires a positive vertex/index count divisible by three");
    case PrimitiveTopology::triangleStrip:
    case PrimitiveTopology::triangleFan:
        return elementCount >= 3
                   ? std::nullopt
                   : std::optional<std::string_view>("requires at least three vertices/indices");
    }
    return "contains an unknown primitive topology";
}

} // namespace

std::string ValidationReport::describe() const {
    if (ok()) {
        return "Scene validation succeeded";
    }
    std::ostringstream stream;
    stream << "Scene validation failed with " << issues.size() << " issue(s):";
    for (const ValidationIssue& validationIssue : issues) {
        stream << "\n - " << validationIssue.path << ": " << validationIssue.message;
    }
    return stream.str();
}

Scene::Scene(std::string name, std::string sourceUri)
    : name_(std::move(name)), sourceUri_(std::move(sourceUri)) {}

ImageHandle Scene::addImage(Image image) {
    return append<ImageHandle>(images_, std::move(image), "image");
}

SamplerHandle Scene::addSampler(Sampler sampler) {
    return append<SamplerHandle>(samplers_, std::move(sampler), "sampler");
}

TextureHandle Scene::addTexture(Texture texture) {
    return append<TextureHandle>(textures_, std::move(texture), "texture");
}

MaterialHandle Scene::addMaterial(Material material) {
    return append<MaterialHandle>(materials_, std::move(material), "material");
}

PrimitiveHandle Scene::addPrimitive(MeshPrimitive primitive) {
    return append<PrimitiveHandle>(primitives_, std::move(primitive), "primitive");
}

MeshHandle Scene::addMesh(Mesh mesh) {
    return append<MeshHandle>(meshes_, std::move(mesh), "mesh");
}

CameraHandle Scene::addCamera(Camera camera) {
    return append<CameraHandle>(cameras_, std::move(camera), "camera");
}

LightHandle Scene::addLight(Light light) {
    return append<LightHandle>(lights_, std::move(light), "light");
}

NodeHandle Scene::addNode(Node node) { return append<NodeHandle>(nodes_, std::move(node), "node"); }

void Scene::addRoot(NodeHandle nodeHandle) {
    (void)node(nodeHandle);
    roots_.push_back(nodeHandle);
}

void Scene::addChild(NodeHandle parent, NodeHandle child) {
    (void)node(child);
    if (!resolves(parent, nodes_.size())) {
        throw std::out_of_range("Invalid parent node handle");
    }
    nodes_[parent.index()].children.push_back(child);
}

const Image& Scene::image(ImageHandle handle) const { return lookup(images_, handle, "image"); }

const Sampler& Scene::sampler(SamplerHandle handle) const {
    return lookup(samplers_, handle, "sampler");
}

const Texture& Scene::texture(TextureHandle handle) const {
    return lookup(textures_, handle, "texture");
}

const Material& Scene::material(MaterialHandle handle) const {
    return lookup(materials_, handle, "material");
}

const MeshPrimitive& Scene::primitive(PrimitiveHandle handle) const {
    return lookup(primitives_, handle, "primitive");
}

const Mesh& Scene::mesh(MeshHandle handle) const { return lookup(meshes_, handle, "mesh"); }

const Camera& Scene::camera(CameraHandle handle) const {
    return lookup(cameras_, handle, "camera");
}

const Light& Scene::light(LightHandle handle) const { return lookup(lights_, handle, "light"); }

const Node& Scene::node(NodeHandle handle) const { return lookup(nodes_, handle, "node"); }

ValidationReport Scene::validate() const {
    ValidationReport report;

    for (std::size_t index = 0; index < images_.size(); ++index) {
        const Image& imageAsset = images_[index];
        if (imageAsset.sourceUri.empty() && imageAsset.encodedData.empty()) {
            issue(report, elementPath("images", index),
                  "must contain encoded data or a source URI");
        }
    }

    for (std::size_t index = 0; index < samplers_.size(); ++index) {
        const Sampler& samplerAsset = samplers_[index];
        const std::string path = elementPath("samplers", index);
        if (!valid(samplerAsset.minFilter) || !valid(samplerAsset.magFilter) ||
            !valid(samplerAsset.mipFilter)) {
            issue(report, path, "contains an unknown filter mode");
        }
        if (!valid(samplerAsset.wrapU) || !valid(samplerAsset.wrapV)) {
            issue(report, path, "contains an unknown wrap mode");
        }
    }

    for (std::size_t index = 0; index < textures_.size(); ++index) {
        const Texture& textureAsset = textures_[index];
        const std::string path = elementPath("textures", index);
        if (!resolves(textureAsset.image, images_.size())) {
            issue(report, path + ".image", "references an image outside this scene");
        }
        if (textureAsset.sampler.valid() && !resolves(textureAsset.sampler, samplers_.size())) {
            issue(report, path + ".sampler", "references a sampler outside this scene");
        }
    }

    for (std::size_t index = 0; index < materials_.size(); ++index) {
        const Material& materialAsset = materials_[index];
        const std::string path = elementPath("materials", index);
        if (!finite(materialAsset.baseColorFactor) ||
            !inRange(materialAsset.baseColorFactor.r, 0.0F, 1.0F) ||
            !inRange(materialAsset.baseColorFactor.g, 0.0F, 1.0F) ||
            !inRange(materialAsset.baseColorFactor.b, 0.0F, 1.0F) ||
            !inRange(materialAsset.baseColorFactor.a, 0.0F, 1.0F)) {
            issue(report, path + ".baseColorFactor", "must be finite and within [0, 1]");
        }
        if (!inRange(materialAsset.metallicFactor, 0.0F, 1.0F)) {
            issue(report, path + ".metallicFactor", "must be finite and within [0, 1]");
        }
        if (!inRange(materialAsset.roughnessFactor, 0.0F, 1.0F)) {
            issue(report, path + ".roughnessFactor", "must be finite and within [0, 1]");
        }
        if (!finite(materialAsset.emissiveFactor) || materialAsset.emissiveFactor.r < 0.0F ||
            materialAsset.emissiveFactor.g < 0.0F || materialAsset.emissiveFactor.b < 0.0F) {
            issue(report, path + ".emissiveFactor", "must be finite and non-negative");
        }
        if (!finite(materialAsset.normalScale)) {
            issue(report, path + ".normalScale", "must be finite");
        }
        if (!inRange(materialAsset.occlusionStrength, 0.0F, 1.0F)) {
            issue(report, path + ".occlusionStrength", "must be finite and within [0, 1]");
        }
        if (!valid(materialAsset.queue)) {
            issue(report, path + ".queue", "contains an unknown material queue");
        }
        if (!inRange(materialAsset.alphaCutoff, 0.0F, 1.0F)) {
            issue(report, path + ".alphaCutoff", "must be finite and within [0, 1]");
        }
        validateBinding(report, materialAsset.baseColorTexture, textures_.size(),
                        path + ".baseColorTexture");
        validateBinding(report, materialAsset.metallicRoughnessTexture, textures_.size(),
                        path + ".metallicRoughnessTexture");
        validateBinding(report, materialAsset.normalTexture, textures_.size(),
                        path + ".normalTexture");
        validateBinding(report, materialAsset.occlusionTexture, textures_.size(),
                        path + ".occlusionTexture");
        validateBinding(report, materialAsset.emissiveTexture, textures_.size(),
                        path + ".emissiveTexture");
    }

    for (std::size_t index = 0; index < primitives_.size(); ++index) {
        const MeshPrimitive& primitiveAsset = primitives_[index];
        const std::string path = elementPath("primitives", index);
        if (primitiveAsset.positions.empty()) {
            issue(report, path + ".positions", "must not be empty");
        }
        validateFiniteElements(report, primitiveAsset.positions, path + ".positions");
        validateFiniteElements(report, primitiveAsset.normals, path + ".normals");
        validateFiniteElements(report, primitiveAsset.tangents, path + ".tangents");
        validateFiniteElements(report, primitiveAsset.texCoords[0], path + ".texCoords[0]");
        validateFiniteElements(report, primitiveAsset.texCoords[1], path + ".texCoords[1]");
        validateFiniteElements(report, primitiveAsset.colors, path + ".colors");

        const auto validateAttributeCount = [&](std::size_t count, std::string_view name) {
            if (count != 0 && count != primitiveAsset.positions.size()) {
                issue(report, path + '.' + std::string(name),
                      "must be empty or match the position count");
            }
        };
        validateAttributeCount(primitiveAsset.normals.size(), "normals");
        validateAttributeCount(primitiveAsset.tangents.size(), "tangents");
        validateAttributeCount(primitiveAsset.texCoords[0].size(), "texCoords[0]");
        validateAttributeCount(primitiveAsset.texCoords[1].size(), "texCoords[1]");
        validateAttributeCount(primitiveAsset.colors.size(), "colors");

        for (std::size_t element = 0; element < primitiveAsset.indices.size(); ++element) {
            if (static_cast<std::size_t>(primitiveAsset.indices[element]) >=
                primitiveAsset.positions.size()) {
                issue(report, path + ".indices[" + std::to_string(element) + ']',
                      "references a vertex outside the position array");
                break;
            }
        }
        const std::size_t drawCount = primitiveAsset.indices.empty()
                                          ? primitiveAsset.positions.size()
                                          : primitiveAsset.indices.size();
        if (const auto error = topologyError(primitiveAsset.topology, drawCount)) {
            issue(report, path + ".topology", std::string(*error));
        }
        if (primitiveAsset.material.valid() &&
            !resolves(primitiveAsset.material, materials_.size())) {
            issue(report, path + ".material", "references a material outside this scene");
        }
    }

    for (std::size_t index = 0; index < meshes_.size(); ++index) {
        const Mesh& meshAsset = meshes_[index];
        const std::string path = elementPath("meshes", index);
        if (meshAsset.primitives.empty()) {
            issue(report, path + ".primitives", "must not be empty");
        }
        std::unordered_set<std::uint32_t> uniquePrimitives;
        for (std::size_t primitiveIndex = 0; primitiveIndex < meshAsset.primitives.size();
             ++primitiveIndex) {
            const PrimitiveHandle primitiveHandle = meshAsset.primitives[primitiveIndex];
            if (!resolves(primitiveHandle, primitives_.size())) {
                issue(report, path + ".primitives[" + std::to_string(primitiveIndex) + ']',
                      "references a primitive outside this scene");
            } else if (!uniquePrimitives.insert(primitiveHandle.index()).second) {
                issue(report, path + ".primitives[" + std::to_string(primitiveIndex) + ']',
                      "duplicates a primitive within the mesh");
            }
        }
    }

    for (std::size_t index = 0; index < cameras_.size(); ++index) {
        const Camera& cameraAsset = cameras_[index];
        const std::string path = elementPath("cameras", index);
        switch (cameraAsset.type) {
        case CameraType::perspective:
            if (!finite(cameraAsset.verticalFieldOfView) ||
                cameraAsset.verticalFieldOfView <= 0.0F ||
                cameraAsset.verticalFieldOfView >= std::numbers::pi_v<float>) {
                issue(report, path + ".verticalFieldOfView",
                      "must be finite and between 0 and pi radians");
            }
            if (cameraAsset.aspectRatio.has_value() &&
                (!finite(*cameraAsset.aspectRatio) || *cameraAsset.aspectRatio <= 0.0F)) {
                issue(report, path + ".aspectRatio", "must be finite and positive");
            }
            if (!finite(cameraAsset.zNear) || cameraAsset.zNear <= 0.0F) {
                issue(report, path + ".zNear", "must be finite and positive");
            }
            if (cameraAsset.zFar.has_value() &&
                (!finite(*cameraAsset.zFar) || *cameraAsset.zFar <= cameraAsset.zNear)) {
                issue(report, path + ".zFar", "must be finite and greater than zNear");
            }
            break;
        case CameraType::orthographic:
            if (!finite(cameraAsset.xMagnitude) || cameraAsset.xMagnitude <= 0.0F ||
                !finite(cameraAsset.yMagnitude) || cameraAsset.yMagnitude <= 0.0F) {
                issue(report, path, "orthographic magnitudes must be finite and positive");
            }
            if (!cameraAsset.zFar.has_value() || !finite(cameraAsset.zNear) ||
                !finite(*cameraAsset.zFar) || *cameraAsset.zFar <= cameraAsset.zNear) {
                issue(report, path, "orthographic zFar must be finite and greater than zNear");
            }
            break;
        default:
            issue(report, path + ".type", "contains an unknown camera type");
            break;
        }
    }

    for (std::size_t index = 0; index < lights_.size(); ++index) {
        const Light& lightAsset = lights_[index];
        const std::string path = elementPath("lights", index);
        if (!finite(lightAsset.color) || lightAsset.color.r < 0.0F || lightAsset.color.g < 0.0F ||
            lightAsset.color.b < 0.0F) {
            issue(report, path + ".color", "must be finite and non-negative");
        }
        if (!finite(lightAsset.intensity) || lightAsset.intensity < 0.0F) {
            issue(report, path + ".intensity", "must be finite and non-negative");
        }
        if (lightAsset.range.has_value() &&
            (!finite(*lightAsset.range) || *lightAsset.range <= 0.0F)) {
            issue(report, path + ".range", "must be finite and positive");
        }
        switch (lightAsset.type) {
        case LightType::directional:
        case LightType::point:
            break;
        case LightType::spot:
            if (!finite(lightAsset.innerConeAngle) || !finite(lightAsset.outerConeAngle) ||
                lightAsset.innerConeAngle < 0.0F ||
                lightAsset.outerConeAngle > std::numbers::pi_v<float> / 2.0F ||
                lightAsset.innerConeAngle > lightAsset.outerConeAngle) {
                issue(report, path, "spot cone angles must satisfy 0 <= inner <= outer <= pi/2");
            }
            break;
        default:
            issue(report, path + ".type", "contains an unknown light type");
            break;
        }
    }

    std::vector<std::vector<std::uint32_t>> validChildren(nodes_.size());
    std::vector<std::uint32_t> parentCounts(nodes_.size(), 0);
    for (std::size_t index = 0; index < nodes_.size(); ++index) {
        const Node& nodeAsset = nodes_[index];
        const std::string path = elementPath("nodes", index);
        if (!finite(nodeAsset.localTransform)) {
            issue(report, path + ".localTransform", "contains a non-finite component");
        }
        if (nodeAsset.mesh.valid() && !resolves(nodeAsset.mesh, meshes_.size())) {
            issue(report, path + ".mesh", "references a mesh outside this scene");
        }
        if (nodeAsset.camera.valid() && !resolves(nodeAsset.camera, cameras_.size())) {
            issue(report, path + ".camera", "references a camera outside this scene");
        }
        if (nodeAsset.light.valid() && !resolves(nodeAsset.light, lights_.size())) {
            issue(report, path + ".light", "references a light outside this scene");
        }

        std::unordered_set<std::uint32_t> uniqueChildren;
        for (std::size_t childIndex = 0; childIndex < nodeAsset.children.size(); ++childIndex) {
            const NodeHandle child = nodeAsset.children[childIndex];
            const std::string childPath = path + ".children[" + std::to_string(childIndex) + ']';
            if (!resolves(child, nodes_.size())) {
                issue(report, childPath, "references a node outside this scene");
                continue;
            }
            if (!uniqueChildren.insert(child.index()).second) {
                issue(report, childPath, "duplicates a child within the node");
                continue;
            }
            validChildren[index].push_back(child.index());
            ++parentCounts[child.index()];
            if (parentCounts[child.index()] > 1) {
                issue(report, childPath, "gives a node more than one parent");
            }
        }
    }

    std::vector<bool> declaredRoot(nodes_.size(), false);
    for (std::size_t index = 0; index < roots_.size(); ++index) {
        const NodeHandle root = roots_[index];
        const std::string path = elementPath("roots", index);
        if (!resolves(root, nodes_.size())) {
            issue(report, path, "references a node outside this scene");
            continue;
        }
        if (declaredRoot[root.index()]) {
            issue(report, path, "duplicates a scene root");
        }
        declaredRoot[root.index()] = true;
        if (parentCounts[root.index()] != 0) {
            issue(report, path, "references a node that also has a parent");
        }
    }
    for (std::size_t index = 0; index < nodes_.size(); ++index) {
        if (parentCounts[index] == 0 && !declaredRoot[index]) {
            issue(report, elementPath("nodes", index), "is parentless but is not a scene root");
        }
    }

    std::vector<std::uint32_t> remainingParents = parentCounts;
    std::deque<std::uint32_t> ready;
    for (std::size_t index = 0; index < remainingParents.size(); ++index) {
        if (remainingParents[index] == 0) {
            ready.push_back(static_cast<std::uint32_t>(index));
        }
    }
    std::size_t visited = 0;
    while (!ready.empty()) {
        const std::uint32_t nodeIndex = ready.front();
        ready.pop_front();
        ++visited;
        for (const std::uint32_t child : validChildren[nodeIndex]) {
            if (--remainingParents[child] == 0) {
                ready.push_back(child);
            }
        }
    }
    if (visited != nodes_.size()) {
        issue(report, "nodes", "hierarchy contains a cycle");
    }

    return report;
}

AxisAlignedBounds computeBounds(const MeshPrimitive& primitive) {
    AxisAlignedBounds bounds;
    if (primitive.positions.empty()) {
        return bounds;
    }
    bounds.minimum = primitive.positions.front();
    bounds.maximum = primitive.positions.front();
    for (const glm::vec3& position : primitive.positions) {
        bounds.minimum.x = std::min(bounds.minimum.x, position.x);
        bounds.minimum.y = std::min(bounds.minimum.y, position.y);
        bounds.minimum.z = std::min(bounds.minimum.z, position.z);
        bounds.maximum.x = std::max(bounds.maximum.x, position.x);
        bounds.maximum.y = std::max(bounds.maximum.y, position.y);
        bounds.maximum.z = std::max(bounds.maximum.z, position.z);
    }
    bounds.valid = true;
    return bounds;
}

SceneEvaluation evaluate(const Scene& scene) {
    const ValidationReport report = scene.validate();
    if (!report.ok()) {
        throw std::runtime_error(report.describe());
    }

    SceneEvaluation evaluation;
    evaluation.nodeWorldTransforms.resize(scene.nodes().size(), glm::mat4(1.0F));
    std::vector<NodeHandle> stack;
    stack.reserve(scene.nodes().size());
    for (auto root = scene.roots().rbegin(); root != scene.roots().rend(); ++root) {
        stack.push_back(*root);
    }

    while (!stack.empty()) {
        const NodeHandle nodeHandle = stack.back();
        stack.pop_back();
        const Node& nodeAsset = scene.node(nodeHandle);
        const glm::mat4 worldTransform =
            evaluation.nodeWorldTransforms[nodeHandle.index()] * nodeAsset.localTransform;
        evaluation.nodeWorldTransforms[nodeHandle.index()] = worldTransform;
        if (nodeAsset.mesh.valid()) {
            evaluation.meshInstances.push_back({.node = nodeHandle,
                                                .mesh = nodeAsset.mesh,
                                                .worldTransform = worldTransform,
                                                .visibilityMask = nodeAsset.visibilityMask});
        }
        for (auto child = nodeAsset.children.rbegin(); child != nodeAsset.children.rend();
             ++child) {
            evaluation.nodeWorldTransforms[child->index()] = worldTransform;
            stack.push_back(*child);
        }
    }
    return evaluation;
}

} // namespace renderer::scene
