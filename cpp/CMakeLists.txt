cmake_minimum_required(VERSION 3.18)
project(seqwin LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

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

execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import pybind11; print(pybind11.get_include())"
    OUTPUT_VARIABLE PYBIND11_INCLUDE
    RESULT_VARIABLE PYBIND11_STATUS
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if (NOT PYBIND11_STATUS EQUAL 0)
    message(FATAL_ERROR "pybind11 Python package is required (pip/conda install pybind11)")
endif()

Python_add_library(_core MODULE WITH_SOABI
    src/bindings/python_bindings.cpp
    src/seqwin/graph.cpp
    src/seqwin/helpers.cpp
    vendor/btllib/minimizer.cpp
    src/seqwin/fasta_reader.cpp
    vendor/btllib/status.cpp
)

target_include_directories(_core PRIVATE include vendor "${PYBIND11_INCLUDE}")
target_link_libraries(_core PRIVATE Python::Module ZLIB::ZLIBSTATIC)

# GCC 8's libstdc++ keeps std::filesystem symbols in a separate library.
# Without this, the extension can compile but fail to import with unresolved
# std::filesystem symbols such as std::filesystem::file_size.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
    target_link_libraries(_core PRIVATE stdc++fs)
endif()

target_compile_definitions(_core PRIVATE PYBIND11_DETAILED_ERROR_MESSAGES)

set_target_properties(_core PROPERTIES PREFIX "")

install(TARGETS _core
    LIBRARY DESTINATION seqwin/graph
    RUNTIME DESTINATION seqwin/graph
    ARCHIVE DESTINATION seqwin/graph
)
