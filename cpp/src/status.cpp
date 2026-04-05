#include "status.hpp"

#include <cerrno>
#include <cstdlib>
#include <iostream>
#include <string>

#ifdef _WIN32
#include <sys/types.h>
#include <sys/stat.h>
#else
#include <sys/stat.h>
#endif

namespace btllib {

std::string
get_time()
{
  time_t now;
  const auto timeret = time(&now);
  if (timeret == (time_t)(-1)) {
    std::cerr << "btllib: time() failed." << std::endl;
    std::exit(EXIT_FAILURE); // NOLINT(concurrency-mt-unsafe)
  }
  char buf[sizeof("2011-10-08T07:07:09Z")];
  std::tm tm_result = {};
#ifdef _WIN32
  localtime_s(&tm_result, &now);
#else
  localtime_r(&now, &tm_result);
#endif
  const auto ret = std::strftime(buf, sizeof(buf), "%F %T", &tm_result);
  if (ret < sizeof(buf) - 2) {
    std::cerr << "btllib: strftime failed." << std::endl;
    std::exit(EXIT_FAILURE); // NOLINT(concurrency-mt-unsafe)
  }
  return std::string(buf);
}

void
log_info(const std::string& msg)
{
  std::string info_msg = "[" + get_time() + "]" + PRINT_COLOR_INFO + "[INFO] " +
                         PRINT_COLOR_END + msg;
  std::cerr << info_msg << std::endl;
}

void
log_warning(const std::string& msg)
{
  std::string warning_msg = "[" + get_time() + "]" + PRINT_COLOR_WARNING +
                            "[WARNING] " + PRINT_COLOR_END + msg;
  std::cerr << warning_msg << std::endl;
}

void
log_error(const std::string& msg)
{
  std::string error_msg = "[" + get_time() + "]" + PRINT_COLOR_ERROR +
                          "[ERROR] " + PRINT_COLOR_END + msg;
  std::cerr << error_msg << std::endl;
}

void
check_info(bool condition, const std::string& msg)
{
  if (condition) {
    log_info(msg);
  }
}

void
check_warning(bool condition, const std::string& msg)
{
  if (condition) {
    log_warning(msg);
  }
}

void
check_error(bool condition, const std::string& msg)
{
  if (condition) {
    log_error(msg);
    std::exit(EXIT_FAILURE); // NOLINT(concurrency-mt-unsafe)
  }
}

std::string
get_strerror()
{
  static const size_t buflen = 1024;
  char buf[buflen];
#ifdef _WIN32
  strerror_s(buf, buflen, errno);
  return buf;
#else
// POSIX and GNU implementation of strerror_r differ, even in function signature
// and so we need to check which one is used
#if __APPLE__ ||                                                               \
  ((_POSIX_C_SOURCE >= 200112L || _XOPEN_SOURCE >= 600) && !_GNU_SOURCE)
  strerror_r(errno, buf, buflen);
  return buf;
#else
  return strerror_r(errno, buf, buflen);
#endif
#endif
}

void
check_stream(const std::ios& stream, const std::string& name)
{
  if (!stream.good()) {
    log_error("'" + name + "' stream error: " + get_strerror());
    std::exit(EXIT_FAILURE); // NOLINT(concurrency-mt-unsafe)
  }
}

void
check_file_accessibility(const std::string& filepath)
{
#ifdef _WIN32
  struct _stat buffer
  {};
  const auto ret = _stat(filepath.c_str(), &buffer);
#else
  struct stat buffer
  {};
  const auto ret = stat(filepath.c_str(), &buffer);
#endif
  btllib::check_error(ret != 0, get_strerror() + ": " + filepath);
}

} // namespace btllib
