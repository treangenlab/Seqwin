#pragma once

#include <pybind11/pybind11.h>

namespace seqwin::bindings {

/**
 * @brief Register native result classes on the Python extension module.
 */
void bind_python_classes(pybind11::module_& module);

} // namespace seqwin::bindings
