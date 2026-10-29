#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "seqwin/graph_view.hpp"

namespace py = pybind11;

namespace seqwin::bindings {

/**
 * @brief Normalize a sequence of paths using Python's `os.fspath()` protocol.
 */
std::vector<std::string> normalize_paths(const py::handle& assembly_paths);

template <typename T>
using NumpyArray = py::array_t<T, py::array::c_style>;

/**
 * @brief Create a read-only, zero-copy NumPy view of a native array.
 *
 * It also keeps `owner` alive to prevent the native buffer from being freed.
 */
template <typename Container>
py::array to_numpy(
    const Container& values,
    const py::handle& owner
) {
    using T = typename Container::value_type;
    static const T empty{};
    py::array result(
        py::dtype::of<T>(),
        {static_cast<py::ssize_t>(values.size())},
        {static_cast<py::ssize_t>(sizeof(T))},
        values.empty() ? &empty : values.data(),
        owner
    );
    result.attr("setflags")(py::arg("write") = false);
    return result;
}

/**
 * @brief Create a non-owning C++ array view over a NumPy array.
 */
template <typename T>
seqwin::ArrayView<T> view_numpy(const py::array& array) {
    return {
        static_cast<const T*>(array.data()),
        static_cast<std::size_t>(array.size())
    };
}

/**
 * @brief Requires a one-dimensional, C-contiguous array with the
 * expected dtype, native byte order, and properly aligned data.
 */
template <typename T>
void validate_numpy(
    const py::array& array,
    const char* name
) {
    const std::string label = std::string("'") + name + "'";

    if (array.ndim() != 1) {
        throw std::invalid_argument(label + " must be one-dimensional");
    }
    if (!(array.flags() & py::array::c_style)) {
        throw std::invalid_argument(label + " must be C-contiguous");
    }
    if (
        !array.dtype().equal(py::dtype::of<T>()) ||
        !array.dtype().attr("isnative").cast<bool>()
    ) {
        throw std::invalid_argument(
            label + " has dtype " + py::str(array.dtype()).cast<std::string>() +
            ", expected " + py::str(py::dtype::of<T>()).cast<std::string>()
        );
    }
    if (reinterpret_cast<std::uintptr_t>(array.data()) % alignof(T) != 0) {
        throw std::invalid_argument(label + " must be aligned");
    }
}

} // namespace seqwin::bindings
