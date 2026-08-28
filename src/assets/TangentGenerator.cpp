#include "renderer/assets/TangentGenerator.hpp"

#include "renderer/scene/Scene.hpp"

#include <mikktspace.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace renderer::assets {
namespace {

struct GenerationContext {
    const scene::MeshPrimitive* primitive = nullptr;
    const std::vector<glm::vec2>* texCoords = nullptr;
    std::vector<glm::vec4>* cornerTangents = nullptr;
};

std::size_t cornerOffset(int face, int vertex) {
    if (face < 0 || vertex < 0 || vertex >= 3) {
        throw std::runtime_error("MikkTSpace requested an invalid triangle corner");
    }
    return static_cast<std::size_t>(face) * 3U + static_cast<std::size_t>(vertex);
}

const GenerationContext& userData(const SMikkTSpaceContext* context) {
    if (context == nullptr || context->m_pUserData == nullptr) {
        throw std::runtime_error("MikkTSpace callback received no mesh context");
    }
    return *static_cast<const GenerationContext*>(context->m_pUserData);
}

std::uint32_t sourceVertex(const GenerationContext& context, int face, int vertex) {
    const std::size_t corner = cornerOffset(face, vertex);
    if (context.primitive->indices.empty()) {
        return static_cast<std::uint32_t>(corner);
    }
    return context.primitive->indices[corner];
}

int getFaceCount(const SMikkTSpaceContext* context) {
    const GenerationContext& data = userData(context);
    const std::size_t cornerCount = data.primitive->indices.empty()
                                        ? data.primitive->positions.size()
                                        : data.primitive->indices.size();
    return static_cast<int>(cornerCount / 3U);
}

int getVerticesOfFace(const SMikkTSpaceContext*, int) { return 3; }

void getPosition(const SMikkTSpaceContext* context, float output[], int face, int vertex) {
    const GenerationContext& data = userData(context);
    const glm::vec3& value = data.primitive->positions[sourceVertex(data, face, vertex)];
    output[0] = value.x;
    output[1] = value.y;
    output[2] = value.z;
}

void getNormal(const SMikkTSpaceContext* context, float output[], int face, int vertex) {
    const GenerationContext& data = userData(context);
    const glm::vec3& value = data.primitive->normals[sourceVertex(data, face, vertex)];
    output[0] = value.x;
    output[1] = value.y;
    output[2] = value.z;
}

void getTexCoord(const SMikkTSpaceContext* context, float output[], int face, int vertex) {
    const GenerationContext& data = userData(context);
    const glm::vec2& value = (*data.texCoords)[sourceVertex(data, face, vertex)];
    output[0] = value.x;
    output[1] = value.y;
}

void setTangent(const SMikkTSpaceContext* context, const float tangent[], float sign, int face,
                int vertex) {
    const GenerationContext& data = userData(context);
    (*data.cornerTangents)[cornerOffset(face, vertex)] =
        glm::vec4(tangent[0], tangent[1], tangent[2], sign);
}

struct VertexKey {
    std::uint32_t source = 0;
    std::array<std::uint32_t, 4> tangent{};

    bool operator==(const VertexKey&) const = default;
};

struct VertexKeyHash {
    std::size_t operator()(const VertexKey& key) const {
        std::size_t hash = key.source;
        for (const std::uint32_t component : key.tangent) {
            hash ^= static_cast<std::size_t>(component) + 0x9e3779b9U + (hash << 6U) +
                    (hash >> 2U);
        }
        return hash;
    }
};

VertexKey makeKey(std::uint32_t source, const glm::vec4& tangent) {
    return {
        .source = source,
        .tangent = {std::bit_cast<std::uint32_t>(tangent.x),
                    std::bit_cast<std::uint32_t>(tangent.y),
                    std::bit_cast<std::uint32_t>(tangent.z),
                    std::bit_cast<std::uint32_t>(tangent.w)},
    };
}

glm::vec3 pbrtCoordinateSystemTangent(glm::vec3 normal) {
    // Derived from pbrt-v4 CoordinateSystem(), Copyright (c) 1998-2020
    // Matt Pharr, Wenzel Jakob, and Greg Humphreys; Apache-2.0. This is only
    // used where degenerate geometry/UVs leave MikkTSpace with no tangent.
    normal = glm::normalize(normal);
    const float sign = std::copysign(1.0F, normal.z);
    const float a = -1.0F / (sign + normal.z);
    const float b = normal.x * normal.y * a;
    return {1.0F + sign * normal.x * normal.x * a, sign * b, -sign * normal.x};
}

template <typename Element>
void appendIfPresent(std::vector<Element>& destination, const std::vector<Element>& source,
                     std::uint32_t sourceIndex) {
    if (!source.empty()) {
        destination.push_back(source[sourceIndex]);
    }
}

void reindexWithCornerTangents(scene::MeshPrimitive& primitive,
                               std::span<const glm::vec4> cornerTangents) {
    const std::vector<std::uint32_t> sourceIndices = primitive.indices;
    const bool originallyIndexed = !sourceIndices.empty();
    if (!originallyIndexed) {
        primitive.tangents.assign(cornerTangents.begin(), cornerTangents.end());
        return;
    }

    scene::MeshPrimitive result;
    result.name = primitive.name;
    result.topology = primitive.topology;
    result.material = primitive.material;
    result.positions.reserve(sourceIndices.size());
    result.normals.reserve(sourceIndices.size());
    result.tangents.reserve(sourceIndices.size());
    result.texCoords[0].reserve(sourceIndices.size());
    result.texCoords[1].reserve(sourceIndices.size());
    result.colors.reserve(sourceIndices.size());
    result.indices.reserve(sourceIndices.size());

    std::unordered_map<VertexKey, std::uint32_t, VertexKeyHash> vertices;
    vertices.reserve(sourceIndices.size());
    for (std::size_t corner = 0; corner < sourceIndices.size(); ++corner) {
        const std::uint32_t sourceIndex = sourceIndices[corner];
        const VertexKey key = makeKey(sourceIndex, cornerTangents[corner]);
        const auto existing = vertices.find(key);
        if (existing != vertices.end()) {
            result.indices.push_back(existing->second);
            continue;
        }
        if (result.positions.size() >= std::numeric_limits<std::uint32_t>::max()) {
            throw std::length_error("MikkTSpace output exceeds the glTF vertex index range");
        }
        const std::uint32_t destinationIndex =
            static_cast<std::uint32_t>(result.positions.size());
        vertices.emplace(key, destinationIndex);
        result.positions.push_back(primitive.positions[sourceIndex]);
        appendIfPresent(result.normals, primitive.normals, sourceIndex);
        result.tangents.push_back(cornerTangents[corner]);
        appendIfPresent(result.texCoords[0], primitive.texCoords[0], sourceIndex);
        appendIfPresent(result.texCoords[1], primitive.texCoords[1], sourceIndex);
        appendIfPresent(result.colors, primitive.colors, sourceIndex);
        result.indices.push_back(destinationIndex);
    }
    primitive = std::move(result);
}

} // namespace

void generateMikkTangents(scene::MeshPrimitive& primitive, std::uint32_t texCoordSet) {
    if (!primitive.tangents.empty()) {
        return;
    }
    if (primitive.topology != scene::PrimitiveTopology::triangles) {
        throw std::runtime_error("MikkTSpace tangent generation requires triangle-list topology");
    }
    if (texCoordSet >= primitive.texCoords.size()) {
        throw std::runtime_error("MikkTSpace tangent generation requested an unsupported UV set");
    }
    const std::size_t vertexCount = primitive.positions.size();
    if (vertexCount == 0 || primitive.normals.size() != vertexCount ||
        primitive.texCoords[texCoordSet].size() != vertexCount) {
        throw std::runtime_error(
            "MikkTSpace tangent generation requires matching positions, normals, and UVs");
    }
    const auto validOptionalAttribute = [vertexCount](std::size_t count) {
        return count == 0 || count == vertexCount;
    };
    if (!validOptionalAttribute(primitive.texCoords[0].size()) ||
        !validOptionalAttribute(primitive.texCoords[1].size()) ||
        !validOptionalAttribute(primitive.colors.size())) {
        throw std::runtime_error(
            "MikkTSpace tangent generation found mismatched optional vertex attributes");
    }
    const std::size_t cornerCount =
        primitive.indices.empty() ? vertexCount : primitive.indices.size();
    if (cornerCount == 0 || cornerCount % 3U != 0 ||
        cornerCount / 3U > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("MikkTSpace tangent generation received an invalid face count");
    }
    for (const std::uint32_t index : primitive.indices) {
        if (index >= vertexCount) {
            throw std::runtime_error("MikkTSpace tangent generation found an invalid index");
        }
    }
    for (const glm::vec3& normal : primitive.normals) {
        const float lengthSquared = glm::dot(normal, normal);
        if (!std::isfinite(lengthSquared) || lengthSquared <= 1.0e-20F) {
            throw std::runtime_error(
                "MikkTSpace tangent generation requires finite nonzero normals");
        }
    }

    std::vector<glm::vec4> cornerTangents(cornerCount);
    GenerationContext generation{
        .primitive = &primitive,
        .texCoords = &primitive.texCoords[texCoordSet],
        .cornerTangents = &cornerTangents,
    };
    SMikkTSpaceInterface interface{
        .m_getNumFaces = getFaceCount,
        .m_getNumVerticesOfFace = getVerticesOfFace,
        .m_getPosition = getPosition,
        .m_getNormal = getNormal,
        .m_getTexCoord = getTexCoord,
        .m_setTSpaceBasic = setTangent,
        .m_setTSpace = nullptr,
    };
    const SMikkTSpaceContext context{
        .m_pInterface = &interface,
        .m_pUserData = &generation,
    };
    if (genTangSpaceDefault(&context) == 0) {
        throw std::runtime_error("MikkTSpace failed to generate a tangent frame");
    }
    for (std::size_t corner = 0; corner < cornerTangents.size(); ++corner) {
        glm::vec4& tangent = cornerTangents[corner];
        const std::uint32_t sourceIndex = primitive.indices.empty()
                                              ? static_cast<std::uint32_t>(corner)
                                              : primitive.indices[corner];
        const glm::vec3 normal = glm::normalize(primitive.normals[sourceIndex]);
        // PBRT-v4 GramSchmidt(): v - Dot(v, w) * w. MikkTSpace normally
        // returns an orthogonal vector; this also cleans up inherited frames
        // on degenerate islands before they cross the asset boundary.
        glm::vec3 orthogonal =
            glm::vec3(tangent) - glm::dot(glm::vec3(tangent), normal) * normal;
        const float lengthSquared = glm::dot(orthogonal, orthogonal);
        const float sign = std::isfinite(tangent.w) && std::abs(tangent.w) > 0.5F
                               ? std::copysign(1.0F, tangent.w)
                               : 1.0F;
        if (!std::isfinite(lengthSquared) || lengthSquared <= 1.0e-20F) {
            orthogonal = pbrtCoordinateSystemTangent(normal);
        } else {
            orthogonal /= std::sqrt(lengthSquared);
        }
        tangent = glm::vec4(orthogonal, sign);
    }
    reindexWithCornerTangents(primitive, cornerTangents);
}

} // namespace renderer::assets
