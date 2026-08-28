#include "renderer/core/Log.hpp"

#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <mutex>

namespace renderer::core {
namespace {

std::mutex logMutex;

constexpr std::string_view label(LogLevel level) {
    switch (level) {
    case LogLevel::info:
        return "INFO";
    case LogLevel::warning:
        return "WARN";
    case LogLevel::error:
        return "ERROR";
    }
    return "UNKNOWN";
}

} // namespace

void writeLog(LogLevel level, std::string_view message) {
    const auto now = std::chrono::system_clock::now();
    const auto time = std::chrono::system_clock::to_time_t(now);
    const auto milliseconds =
        std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

    std::tm localTime{};
    localtime_r(&time, &localTime);

    std::scoped_lock lock(logMutex);
    auto& output = level == LogLevel::error ? std::cerr : std::clog;
    output << std::put_time(&localTime, "%H:%M:%S") << '.' << std::setfill('0') << std::setw(3)
           << milliseconds.count() << " [" << label(level) << "] " << message << '\n';
}

} // namespace renderer::core
