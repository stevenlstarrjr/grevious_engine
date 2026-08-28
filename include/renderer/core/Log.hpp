#pragma once

#include <sstream>
#include <string_view>
#include <utility>

namespace renderer::core {

enum class LogLevel { info, warning, error };

void writeLog(LogLevel level, std::string_view message);

template <typename... Args> void log(LogLevel level, Args&&... args) {
    std::ostringstream stream;
    (stream << ... << std::forward<Args>(args));
    writeLog(level, stream.str());
}

} // namespace renderer::core
