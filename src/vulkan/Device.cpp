#include "renderer/vulkan/Device.hpp"

#include "renderer/core/Log.hpp"
#include "renderer/vulkan/Instance.hpp"
#include "renderer/vulkan/VulkanError.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace renderer::vulkan {
namespace {

struct DeviceFeatures {
    VkPhysicalDeviceFeatures2 base{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    VkPhysicalDeviceVulkan12Features vulkan12{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan13Features vulkan13{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceAccelerationStructureFeaturesKHR acceleration{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rayPipeline{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
    VkPhysicalDeviceRayQueryFeaturesKHR rayQuery{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
    VkPhysicalDeviceMeshShaderFeaturesEXT meshShader{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};
};

struct Candidate {
    VkPhysicalDevice handle = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties properties{};
    QueueFamilies queues{};
    OptionalCapabilities optional{};
    int score = std::numeric_limits<int>::min();
};

std::vector<VkExtensionProperties> deviceExtensions(VkPhysicalDevice device) {
    std::uint32_t count = 0;
    check(vkEnumerateDeviceExtensionProperties(device, nullptr, &count, nullptr),
          "vkEnumerateDeviceExtensionProperties(count)");
    std::vector<VkExtensionProperties> extensions(count);
    check(vkEnumerateDeviceExtensionProperties(device, nullptr, &count, extensions.data()),
          "vkEnumerateDeviceExtensionProperties");
    return extensions;
}

bool hasExtension(const std::vector<VkExtensionProperties>& extensions, const char* name) {
    return std::ranges::any_of(extensions, [name](const auto& extension) {
        return std::strcmp(extension.extensionName, name) == 0;
    });
}

std::optional<QueueFamilies> findQueueFamilies(VkPhysicalDevice device, VkSurfaceKHR surface) {
    std::uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, nullptr);
    std::vector<VkQueueFamilyProperties> properties(count);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, properties.data());

    QueueFamilies families{};
    for (std::uint32_t index = 0; index < count; ++index) {
        if (properties[index].queueCount == 0) {
            continue;
        }
        VkBool32 presentSupported = VK_FALSE;
        check(vkGetPhysicalDeviceSurfaceSupportKHR(device, index, surface, &presentSupported),
              "vkGetPhysicalDeviceSurfaceSupportKHR");
        if ((properties[index].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0 &&
            families.graphics == VK_QUEUE_FAMILY_IGNORED) {
            families.graphics = index;
        }
        if (presentSupported == VK_TRUE && families.present == VK_QUEUE_FAMILY_IGNORED) {
            families.present = index;
        }
        if ((properties[index].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0 &&
            (properties[index].queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0) {
            families.compute = index;
        }
        if ((properties[index].queueFlags & VK_QUEUE_TRANSFER_BIT) != 0 &&
            (properties[index].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) == 0) {
            families.transfer = index;
        }
    }

    if (families.graphics == VK_QUEUE_FAMILY_IGNORED ||
        families.present == VK_QUEUE_FAMILY_IGNORED) {
        return std::nullopt;
    }
    if (families.compute == VK_QUEUE_FAMILY_IGNORED) {
        families.compute = families.graphics;
    }
    if (families.transfer == VK_QUEUE_FAMILY_IGNORED) {
        families.transfer = families.compute;
    }
    return families;
}

bool hasSwapchainSupport(VkPhysicalDevice device, VkSurfaceKHR surface) {
    std::uint32_t formatCount = 0;
    std::uint32_t presentModeCount = 0;
    check(vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr),
          "vkGetPhysicalDeviceSurfaceFormatsKHR(count)");
    check(vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr),
          "vkGetPhysicalDeviceSurfacePresentModesKHR(count)");
    return formatCount > 0 && presentModeCount > 0;
}

void queryFeatures(VkPhysicalDevice device, const std::vector<VkExtensionProperties>& extensions,
                   DeviceFeatures& features) {
    features.base.pNext = &features.vulkan12;
    features.vulkan12.pNext = &features.vulkan13;

    void** tail = &features.vulkan13.pNext;
    if (hasExtension(extensions, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME)) {
        *tail = &features.acceleration;
        tail = &features.acceleration.pNext;
    }
    if (hasExtension(extensions, VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME)) {
        *tail = &features.rayPipeline;
        tail = &features.rayPipeline.pNext;
    }
    if (hasExtension(extensions, VK_KHR_RAY_QUERY_EXTENSION_NAME)) {
        *tail = &features.rayQuery;
        tail = &features.rayQuery.pNext;
    }
    if (hasExtension(extensions, VK_EXT_MESH_SHADER_EXTENSION_NAME)) {
        *tail = &features.meshShader;
    }
    vkGetPhysicalDeviceFeatures2(device, &features.base);
}

bool requiredFeaturesAvailable(const DeviceFeatures& features) {
    const auto& base = features.base.features;
    const auto& v12 = features.vulkan12;
    const auto& v13 = features.vulkan13;
    return base.multiDrawIndirect == VK_TRUE && base.samplerAnisotropy == VK_TRUE &&
           v12.drawIndirectCount == VK_TRUE && v12.descriptorIndexing == VK_TRUE &&
           v12.shaderSampledImageArrayNonUniformIndexing == VK_TRUE &&
           v12.descriptorBindingSampledImageUpdateAfterBind == VK_TRUE &&
           v12.descriptorBindingPartiallyBound == VK_TRUE &&
           v12.descriptorBindingVariableDescriptorCount == VK_TRUE &&
           v12.runtimeDescriptorArray == VK_TRUE && v12.timelineSemaphore == VK_TRUE &&
           v12.bufferDeviceAddress == VK_TRUE && v13.synchronization2 == VK_TRUE &&
           v13.dynamicRendering == VK_TRUE && v13.shaderDemoteToHelperInvocation == VK_TRUE;
}

OptionalCapabilities optionalCapabilities(const DeviceFeatures& features,
                                          const std::vector<VkExtensionProperties>& extensions) {
    const bool deferred = hasExtension(extensions, VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
    const bool acceleration =
        deferred && hasExtension(extensions, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME) &&
        features.acceleration.accelerationStructure == VK_TRUE;
    return {
        .accelerationStructure = acceleration,
        .rayTracingPipeline =
            acceleration && hasExtension(extensions, VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME) &&
            features.rayPipeline.rayTracingPipeline == VK_TRUE,
        .rayQuery = acceleration && hasExtension(extensions, VK_KHR_RAY_QUERY_EXTENSION_NAME) &&
                    features.rayQuery.rayQuery == VK_TRUE,
        .meshShader = hasExtension(extensions, VK_EXT_MESH_SHADER_EXTENSION_NAME) &&
                      features.meshShader.meshShader == VK_TRUE,
    };
}

void queryOptionalProperties(VkPhysicalDevice device, OptionalCapabilities& capabilities) {
    VkPhysicalDeviceAccelerationStructurePropertiesKHR acceleration{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR rayPipeline{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
    VkPhysicalDeviceProperties2 properties{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};

    void** tail = &properties.pNext;
    if (capabilities.accelerationStructure) {
        *tail = &acceleration;
        tail = &acceleration.pNext;
    }
    if (capabilities.rayTracingPipeline) {
        *tail = &rayPipeline;
    }
    vkGetPhysicalDeviceProperties2(device, &properties);

    if (capabilities.accelerationStructure) {
        capabilities.maxAccelerationStructureGeometryCount = acceleration.maxGeometryCount;
        capabilities.maxAccelerationStructurePrimitiveCount = acceleration.maxPrimitiveCount;
        capabilities.maxAccelerationStructureInstanceCount = acceleration.maxInstanceCount;
        capabilities.minAccelerationStructureScratchAlignment =
            acceleration.minAccelerationStructureScratchOffsetAlignment;
    }
    if (capabilities.rayTracingPipeline) {
        capabilities.maxRayRecursionDepth = rayPipeline.maxRayRecursionDepth;
        capabilities.shaderGroupHandleSize = rayPipeline.shaderGroupHandleSize;
        capabilities.shaderGroupHandleAlignment = rayPipeline.shaderGroupHandleAlignment;
        capabilities.shaderGroupBaseAlignment = rayPipeline.shaderGroupBaseAlignment;
        capabilities.maxShaderGroupStride = rayPipeline.maxShaderGroupStride;
    }
}

int deviceScore(const VkPhysicalDeviceProperties& properties, const QueueFamilies& queues) {
    int score = 0;
    switch (properties.deviceType) {
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
        score += 40'000;
        break;
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
        score += 30'000;
        break;
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
        score += 20'000;
        break;
    case VK_PHYSICAL_DEVICE_TYPE_CPU:
        score += 10'000;
        break;
    default:
        break;
    }
    score += static_cast<int>(std::min(properties.limits.maxImageDimension2D, 16'384U));
    score += queues.compute != queues.graphics ? 1'000 : 0;
    score += queues.transfer != queues.graphics && queues.transfer != queues.compute ? 500 : 0;
    return score;
}

std::optional<Candidate> inspectCandidate(VkPhysicalDevice device, VkSurfaceKHR surface) {
    Candidate candidate{.handle = device};
    vkGetPhysicalDeviceProperties(device, &candidate.properties);
    if (candidate.properties.apiVersion < VK_API_VERSION_1_3) {
        return std::nullopt;
    }

    const auto extensions = deviceExtensions(device);
    if (!hasExtension(extensions, VK_KHR_SWAPCHAIN_EXTENSION_NAME) ||
        !hasSwapchainSupport(device, surface)) {
        return std::nullopt;
    }
    const auto queues = findQueueFamilies(device, surface);
    if (!queues.has_value()) {
        return std::nullopt;
    }
    DeviceFeatures features;
    queryFeatures(device, extensions, features);
    if (!requiredFeaturesAvailable(features)) {
        return std::nullopt;
    }

    candidate.queues = *queues;
    candidate.optional = optionalCapabilities(features, extensions);
    queryOptionalProperties(device, candidate.optional);
    candidate.score = deviceScore(candidate.properties, candidate.queues);
    return candidate;
}

} // namespace

Device::Device(const Instance& instance) {
    instanceHandle_ = instance.handle();
    std::uint32_t count = 0;
    check(vkEnumeratePhysicalDevices(instance.handle(), &count, nullptr),
          "vkEnumeratePhysicalDevices(count)");
    if (count == 0) {
        throw std::runtime_error("No Vulkan physical devices were found");
    }
    std::vector<VkPhysicalDevice> devices(count);
    check(vkEnumeratePhysicalDevices(instance.handle(), &count, devices.data()),
          "vkEnumeratePhysicalDevices");

    std::optional<Candidate> selected;
    for (VkPhysicalDevice device : devices) {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(device, &properties);
        const auto candidate = inspectCandidate(device, instance.surface());
        if (!candidate.has_value()) {
            core::log(core::LogLevel::warning,
                      "Skipping incompatible Vulkan device: ", properties.deviceName);
            continue;
        }
        if (!selected.has_value() || candidate->score > selected->score) {
            selected = candidate;
        }
    }
    if (!selected.has_value()) {
        throw std::runtime_error(
            "No device satisfies the Vulkan 1.3, swapchain, queue, and required feature policy");
    }

    physicalDevice_ = selected->handle;
    queueFamilies_ = selected->queues;
    supportedOptionalCapabilities_ = selected->optional;
    enabledOptionalCapabilities_ = supportedOptionalCapabilities_;
    // Mesh shaders have no live execution backend yet. Unlike the M8 ray-tracing
    // foundation, do not enable an optional device feature before its owner exists.
    enabledOptionalCapabilities_.meshShader = false;

    const std::set<std::uint32_t> uniqueFamilies{queueFamilies_.graphics, queueFamilies_.present,
                                                 queueFamilies_.compute, queueFamilies_.transfer};
    constexpr float priority = 1.0F;
    std::vector<VkDeviceQueueCreateInfo> queueInfos;
    queueInfos.reserve(uniqueFamilies.size());
    for (const std::uint32_t family : uniqueFamilies) {
        queueInfos.push_back({
            .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .queueFamilyIndex = family,
            .queueCount = 1,
            .pQueuePriorities = &priority,
        });
    }

    VkPhysicalDeviceVulkan13Features vulkan13{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
        .shaderDemoteToHelperInvocation = VK_TRUE,
        .synchronization2 = VK_TRUE,
        .dynamicRendering = VK_TRUE,
    };
    VkPhysicalDeviceVulkan12Features vulkan12{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        .pNext = &vulkan13,
        .drawIndirectCount = VK_TRUE,
        .descriptorIndexing = VK_TRUE,
        .shaderSampledImageArrayNonUniformIndexing = VK_TRUE,
        .descriptorBindingSampledImageUpdateAfterBind = VK_TRUE,
        .descriptorBindingPartiallyBound = VK_TRUE,
        .descriptorBindingVariableDescriptorCount = VK_TRUE,
        .runtimeDescriptorArray = VK_TRUE,
        .timelineSemaphore = VK_TRUE,
        .bufferDeviceAddress = VK_TRUE,
    };
    VkPhysicalDeviceAccelerationStructureFeaturesKHR acceleration{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
        .accelerationStructure = enabledOptionalCapabilities_.accelerationStructure,
    };
    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rayPipeline{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR,
        .rayTracingPipeline = enabledOptionalCapabilities_.rayTracingPipeline,
    };
    VkPhysicalDeviceRayQueryFeaturesKHR rayQuery{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR,
        .rayQuery = enabledOptionalCapabilities_.rayQuery,
    };
    void** optionalFeatureTail = &vulkan13.pNext;
    if (enabledOptionalCapabilities_.accelerationStructure) {
        *optionalFeatureTail = &acceleration;
        optionalFeatureTail = &acceleration.pNext;
    }
    if (enabledOptionalCapabilities_.rayTracingPipeline) {
        *optionalFeatureTail = &rayPipeline;
        optionalFeatureTail = &rayPipeline.pNext;
    }
    if (enabledOptionalCapabilities_.rayQuery) {
        *optionalFeatureTail = &rayQuery;
    }
    VkPhysicalDeviceFeatures2 features{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        .pNext = &vulkan12,
        .features =
            {
                .multiDrawIndirect = VK_TRUE,
                .samplerAnisotropy = VK_TRUE,
            },
    };
    std::vector<const char*> extensions{VK_KHR_SWAPCHAIN_EXTENSION_NAME};
    if (enabledOptionalCapabilities_.accelerationStructure) {
        extensions.push_back(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
        extensions.push_back(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
    }
    if (enabledOptionalCapabilities_.rayTracingPipeline) {
        extensions.push_back(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
    }
    if (enabledOptionalCapabilities_.rayQuery) {
        extensions.push_back(VK_KHR_RAY_QUERY_EXTENSION_NAME);
    }
    const VkDeviceCreateInfo createInfo{
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = &features,
        .queueCreateInfoCount = static_cast<std::uint32_t>(queueInfos.size()),
        .pQueueCreateInfos = queueInfos.data(),
        .enabledExtensionCount = static_cast<std::uint32_t>(extensions.size()),
        .ppEnabledExtensionNames = extensions.data(),
    };
    check(vkCreateDevice(physicalDevice_, &createInfo, nullptr, &device_), "vkCreateDevice");
    volkLoadDevice(device_);

    vkGetDeviceQueue(device_, queueFamilies_.graphics, 0, &graphicsQueue_);
    vkGetDeviceQueue(device_, queueFamilies_.present, 0, &presentQueue_);
    vkGetDeviceQueue(device_, queueFamilies_.compute, 0, &computeQueue_);
    vkGetDeviceQueue(device_, queueFamilies_.transfer, 0, &transferQueue_);

    const auto& properties = selected->properties;
    core::log(core::LogLevel::info, "Selected GPU: ", properties.deviceName, " (Vulkan ",
              VK_API_VERSION_MAJOR(properties.apiVersion), '.',
              VK_API_VERSION_MINOR(properties.apiVersion), '.',
              VK_API_VERSION_PATCH(properties.apiVersion), ')');
    core::log(core::LogLevel::info, "Queue families: graphics=", queueFamilies_.graphics,
              ", present=", queueFamilies_.present, ", compute=", queueFamilies_.compute,
              ", transfer=", queueFamilies_.transfer);
    core::log(core::LogLevel::info,
              "Optional GPU support: AS=", supportedOptionalCapabilities_.accelerationStructure,
              ", RT pipeline=", supportedOptionalCapabilities_.rayTracingPipeline,
              ", ray query=", supportedOptionalCapabilities_.rayQuery,
              ", mesh shader=", supportedOptionalCapabilities_.meshShader);
    core::log(core::LogLevel::info, "Optional device features enabled: AS=",
              enabledOptionalCapabilities_.accelerationStructure,
              ", RT pipeline=", enabledOptionalCapabilities_.rayTracingPipeline,
              ", ray query=", enabledOptionalCapabilities_.rayQuery,
              ", mesh shader=", enabledOptionalCapabilities_.meshShader);
    if (supportedOptionalCapabilities_.accelerationStructure) {
        core::log(core::LogLevel::info, "Acceleration-structure limits: ",
                  supportedOptionalCapabilities_.maxAccelerationStructureInstanceCount,
                  " TLAS instances, ",
                  supportedOptionalCapabilities_.maxAccelerationStructurePrimitiveCount,
                  " BLAS primitives, ",
                  supportedOptionalCapabilities_.maxAccelerationStructureGeometryCount,
                  " geometries; scratch alignment ",
                  supportedOptionalCapabilities_.minAccelerationStructureScratchAlignment,
                  " bytes");
    }
    if (supportedOptionalCapabilities_.rayTracingPipeline) {
        core::log(core::LogLevel::info, "Ray-pipeline limits: recursion depth ",
                  supportedOptionalCapabilities_.maxRayRecursionDepth, ", shader-group handle ",
                  supportedOptionalCapabilities_.shaderGroupHandleSize, " bytes, alignments ",
                  supportedOptionalCapabilities_.shaderGroupHandleAlignment, '/',
                  supportedOptionalCapabilities_.shaderGroupBaseAlignment, " bytes; maximum stride ",
                  supportedOptionalCapabilities_.maxShaderGroupStride, " bytes");
    }
}

Device::~Device() {
    if (device_ != VK_NULL_HANDLE) {
        vkDestroyDevice(device_, nullptr);
    }
}

} // namespace renderer::vulkan
