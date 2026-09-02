#include "renderer/vulkan/Instance.hpp"

#include "renderer/core/Log.hpp"
#include "renderer/platform/RenderSurface.hpp"
#include "renderer/vulkan/VulkanError.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace renderer::vulkan {
namespace {

constexpr const char* validationLayer = "VK_LAYER_KHRONOS_validation";

class VolkInitializationGuard final {
  public:
    explicit VolkInitializationGuard(PFN_vkGetInstanceProcAddr customGetInstanceProcAddr) {
        if (customGetInstanceProcAddr != nullptr) {
            volkInitializeCustom(customGetInstanceProcAddr);
        } else {
            check(volkInitialize(), "volkInitialize");
        }
    }
    ~VolkInitializationGuard() {
        if (active_) {
            volkFinalize();
        }
    }

    VolkInitializationGuard(const VolkInitializationGuard&) = delete;
    VolkInitializationGuard& operator=(const VolkInitializationGuard&) = delete;

    void release() { active_ = false; }

  private:
    bool active_ = true;
};

bool containsExtension(const std::vector<VkExtensionProperties>& extensions, const char* name) {
    return std::ranges::any_of(extensions, [name](const auto& extension) {
        return std::strcmp(extension.extensionName, name) == 0;
    });
}

std::vector<VkExtensionProperties> enumerateExtensions(const char* layerName) {
    std::uint32_t count = 0;
    check(vkEnumerateInstanceExtensionProperties(layerName, &count, nullptr),
          "vkEnumerateInstanceExtensionProperties(count)");
    std::vector<VkExtensionProperties> extensions(count);
    check(vkEnumerateInstanceExtensionProperties(layerName, &count, extensions.data()),
          "vkEnumerateInstanceExtensionProperties");
    return extensions;
}

bool hasLayer(const char* name) {
    std::uint32_t count = 0;
    check(vkEnumerateInstanceLayerProperties(&count, nullptr),
          "vkEnumerateInstanceLayerProperties(count)");
    std::vector<VkLayerProperties> layers(count);
    check(vkEnumerateInstanceLayerProperties(&count, layers.data()),
          "vkEnumerateInstanceLayerProperties");
    return std::ranges::any_of(
        layers, [name](const auto& layer) { return std::strcmp(layer.layerName, name) == 0; });
}

VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                             VkDebugUtilsMessageTypeFlagsEXT,
                                             const VkDebugUtilsMessengerCallbackDataEXT* data,
                                             void* userData) {
    const auto level =
        severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT     ? core::LogLevel::error
        : severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT ? core::LogLevel::warning
                                                                      : core::LogLevel::info;
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT && userData != nullptr) {
        static_cast<std::atomic_uint32_t*>(userData)->fetch_add(1, std::memory_order_relaxed);
    }
    core::log(level, "Vulkan: ",
              data != nullptr && data->pMessage != nullptr ? data->pMessage : "<no message>");
    return VK_FALSE;
}

VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo(void* userData) {
    VkDebugUtilsMessengerCreateInfoEXT info{
        .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
        .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
        .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
        .pfnUserCallback = debugCallback,
        .pUserData = userData,
    };
    return info;
}

} // namespace

Instance::Instance(const platform::RenderSurface& surfaceHost, ValidationMode validationMode,
                   PFN_vkGetInstanceProcAddr customGetInstanceProcAddr)
    : surfaceHost_(surfaceHost) {
    VolkInitializationGuard volkGuard(customGetInstanceProcAddr);
    std::uint32_t loaderVersion = VK_API_VERSION_1_0;
    check(vkEnumerateInstanceVersion(&loaderVersion), "vkEnumerateInstanceVersion");
    if (loaderVersion < VK_API_VERSION_1_3) {
        throw std::runtime_error("Vulkan 1.3 loader support is required");
    }

    const auto availableExtensions = enumerateExtensions(nullptr);

    validationEnabled_ = validationMode != ValidationMode::off && hasLayer(validationLayer);
    if (validationMode == ValidationMode::required && !validationEnabled_) {
        throw std::runtime_error("VK_LAYER_KHRONOS_validation was required but is not installed");
    }
    if (validationMode == ValidationMode::available && !validationEnabled_) {
        core::log(core::LogLevel::warning,
                  "VK_LAYER_KHRONOS_validation is unavailable; continuing without validation");
    }

    const auto layerExtensions = validationEnabled_ ? enumerateExtensions(validationLayer)
                                                    : std::vector<VkExtensionProperties>{};
    const auto extensionAvailable = [&](const char* name) {
        return containsExtension(availableExtensions, name) ||
               containsExtension(layerExtensions, name);
    };

    auto enabledExtensions = surfaceHost.requiredVulkanExtensions();
    const bool debugUtilsAvailable = extensionAvailable(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    if (debugUtilsAvailable) {
        enabledExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }
    const bool validationFeaturesAvailable =
        validationEnabled_ && extensionAvailable(VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME);
    if (validationFeaturesAvailable) {
        enabledExtensions.push_back(VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME);
    }

    std::ranges::sort(enabledExtensions,
                      [](const char* lhs, const char* rhs) { return std::strcmp(lhs, rhs) < 0; });
    enabledExtensions.erase(std::ranges::unique(enabledExtensions,
                                                [](const char* lhs, const char* rhs) {
                                                    return std::strcmp(lhs, rhs) == 0;
                                                })
                                .begin(),
                            enabledExtensions.end());

    for (const char* required : enabledExtensions) {
        if (!extensionAvailable(required)) {
            throw std::runtime_error(std::string("Required instance extension is unavailable: ") +
                                     required);
        }
    }

    const VkApplicationInfo appInfo{
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "Genesis Render",
        .applicationVersion = VK_MAKE_API_VERSION(0, 0, 1, 0),
        .pEngineName = "Genesis Render",
        .engineVersion = VK_MAKE_API_VERSION(0, 0, 1, 0),
        .apiVersion = VK_API_VERSION_1_3,
    };

    auto debugInfo = debugCreateInfo(&validationErrorCount_);
    constexpr std::array validationFeatureEnables{
        VK_VALIDATION_FEATURE_ENABLE_BEST_PRACTICES_EXT,
        VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT,
    };
    VkValidationFeaturesEXT validationFeatures{
        .sType = VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT,
        .pNext = debugUtilsAvailable ? &debugInfo : nullptr,
        .enabledValidationFeatureCount =
            static_cast<std::uint32_t>(validationFeatureEnables.size()),
        .pEnabledValidationFeatures = validationFeatureEnables.data(),
    };
    const std::vector<const char*> layers =
        validationEnabled_ ? std::vector<const char*>{validationLayer} : std::vector<const char*>{};
    const VkInstanceCreateInfo createInfo{
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pNext = validationFeaturesAvailable ? static_cast<const void*>(&validationFeatures)
                 : debugUtilsAvailable       ? static_cast<const void*>(&debugInfo)
                                             : nullptr,
        .pApplicationInfo = &appInfo,
        .enabledLayerCount = static_cast<std::uint32_t>(layers.size()),
        .ppEnabledLayerNames = layers.data(),
        .enabledExtensionCount = static_cast<std::uint32_t>(enabledExtensions.size()),
        .ppEnabledExtensionNames = enabledExtensions.data(),
    };
    check(vkCreateInstance(&createInfo, nullptr, &instance_), "vkCreateInstance");
    volkLoadInstance(instance_);

    try {
        if (debugUtilsAvailable) {
            const auto createMessenger = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
                vkGetInstanceProcAddr(instance_, "vkCreateDebugUtilsMessengerEXT"));
            if (createMessenger != nullptr) {
                check(createMessenger(instance_, &debugInfo, nullptr, &debugMessenger_),
                      "vkCreateDebugUtilsMessengerEXT");
            }
        }
        surface_ = surfaceHost.createSurface(instance_);
    } catch (...) {
        if (debugMessenger_ != VK_NULL_HANDLE) {
            const auto destroyMessenger = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
                vkGetInstanceProcAddr(instance_, "vkDestroyDebugUtilsMessengerEXT"));
            if (destroyMessenger != nullptr) {
                destroyMessenger(instance_, debugMessenger_, nullptr);
            }
        }
        vkDestroyInstance(instance_, nullptr);
        instance_ = VK_NULL_HANDLE;
        debugMessenger_ = VK_NULL_HANDLE;
        throw;
    }

    core::log(core::LogLevel::info, "Vulkan instance created (loader ",
              VK_API_VERSION_MAJOR(loaderVersion), '.', VK_API_VERSION_MINOR(loaderVersion), '.',
              VK_API_VERSION_PATCH(loaderVersion), ", validation ",
              validationEnabled_ ? "enabled" : "disabled", ')');
    if (validationFeaturesAvailable) {
        core::log(core::LogLevel::info,
                  "Validation features: core, best practices, synchronization");
    }
    volkGuard.release();
}

Instance::~Instance() {
    if (surface_ != VK_NULL_HANDLE && instance_ != VK_NULL_HANDLE) {
        surfaceHost_.destroySurface(instance_, surface_);
    }
    if (debugMessenger_ != VK_NULL_HANDLE && instance_ != VK_NULL_HANDLE) {
        const auto destroyMessenger = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance_, "vkDestroyDebugUtilsMessengerEXT"));
        if (destroyMessenger != nullptr) {
            destroyMessenger(instance_, debugMessenger_, nullptr);
        }
    }
    if (instance_ != VK_NULL_HANDLE) {
        vkDestroyInstance(instance_, nullptr);
    }
    volkFinalize();
}

} // namespace renderer::vulkan
