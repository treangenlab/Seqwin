cmake_minimum_required(VERSION 3.18)
project(minimizer_btllib LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

# Fetch and build zlib from source
include(FetchContent)
FetchContent_Declare(
    zlib
    GIT_REPOSITORY https://github.com/madler/zlib.git
    GIT_TAG        v1.3.2
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
  src/python_bindings.cpp
  src/minimizer.cpp
  src/fasta_reader.cpp
  src/status.cpp
)

target_include_directories(_core PRIVATE include "${PYBIND11_INCLUDE}")
target_link_libraries(_core PRIVATE Python::Module zlibstatic)
target_compile_definitions(_core PRIVATE PYBIND11_DETAILED_ERROR_MESSAGES)

set_target_properties(_core PROPERTIES PREFIX "")

install(TARGETS _core
  LIBRARY DESTINATION seqwin/btllib
  RUNTIME DESTINATION seqwin/btllib
  ARCHIVE DESTINATION seqwin/btllib
)
