#pragma once

#include <string>

namespace seqwin::internal {

/**
 * @brief Emit a message through Python's logging module.
 *
 * @param message Message to log.
 * @param level Logging level: `debug`, `info`, `warning`, `error`, or `critical`.
 */
void log_python(
    const std::string& message,
    const std::string& level = "info"
);

} // namespace seqwin::internal
