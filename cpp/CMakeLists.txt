cmake_minimum_required(VERSION 3.18)
project(seqwin LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

# Fetch and build zlib from source for internal use only.
# Seqwin links zlib statically into the Python extension, so users do not need
# a separate zlib installation at build or runtime.
include(FetchContent)
set(ZLIB_BUILD_SHARED OFF CACHE BOOL "Build zlib shared library" FORCE)
set(ZLIB_BUILD_STATIC ON CACHE BOOL "Build zlib static library" FORCE)
set(ZLIB_BUILD_TESTING OFF CACHE BOOL "Build zlib tests" FORCE)
set(ZLIB_INSTALL OFF CACHE BOOL "Install zlib upstream artifacts" FORCE)
set(SEQWIN_ZLIB_VERSION 1.3.2)
set(SEQWIN_ZLIB_SHA256 bb329a0a2cd0274d05519d61c667c062e06990d72e125ee2dfa8de64f0119d16)
FetchContent_Declare(
    zlib
    URL
        "https://zlib.net/zlib-${SEQWIN_ZLIB_VERSION}.tar.gz"
        "https://zlib.net/fossils/zlib-${SEQWIN_ZLIB_VERSION}.tar.gz"
        "https://github.com/madler/zlib/releases/download/v${SEQWIN_ZLIB_VERSION}/zlib-${SEQWIN_ZLIB_VERSION}.tar.gz"
    URL_HASH SHA256=${SEQWIN_ZLIB_SHA256}
    TIMEOUT 60
)
FetchContent_MakeAvailable(zlib)

pybind11_add_module(_native
    src/bindings/python_graph.cpp
    src/bindings/python_classes.cpp
    src/bindings/python_bindings.cpp
    src/seqwin/build.cpp
    src/seqwin/build_internals.cpp
    src/seqwin/extract.cpp
    src/seqwin/filter_internals.cpp
    src/seqwin/filter.cpp
    src/utils/fasta_reader.cpp
    src/utils/logging.cpp
    vendor/btllib/minimizer.cpp
    vendor/btllib/status.cpp
)

target_include_directories(_native PRIVATE include src vendor)
target_link_libraries(_native PRIVATE ZLIB::ZLIBSTATIC)

# GCC 8's libstdc++ keeps std::filesystem symbols in a separate library.
# Without this, the extension can compile but fail to import with unresolved
# std::filesystem symbols such as std::filesystem::file_size.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
    target_link_libraries(_native PRIVATE stdc++fs)
endif()

target_compile_definitions(_native PRIVATE PYBIND11_DETAILED_ERROR_MESSAGES)

install(TARGETS _native
    LIBRARY DESTINATION seqwin/core
    RUNTIME DESTINATION seqwin/core
    ARCHIVE DESTINATION seqwin/core
)
