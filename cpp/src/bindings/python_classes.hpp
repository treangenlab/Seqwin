#pragma once

#include <variant>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "seqwin/graph_view.hpp"

namespace seqwin::bindings {

/** Python module path used by public classes. */
inline constexpr char public_module[] = "seqwin.core";

/**
 * @brief NumPy arrays representing graph nodes.
 *
 * Serves two roles:
 * 1. Node storage within `GraphNumpy`.
 * 2. Python-exposed `Nodes` class for both storage backends.
 * For native storage, arrays are NumPy views of native buffers.
 * For NumPy-backed storage, they reference the original NumPy arrays.
 */
struct NodesNumpy {
    pybind11::array kmers;
    pybind11::array offsets;
    pybind11::array prevalences;
    pybind11::array hashes;

    std::size_t size() const noexcept {
        return hashes.size();
    }
};

/**
 * @brief Graph storage backed by user-provided NumPy arrays.
 *
 * Arrays may be ordinary arrays or memory-mapped arrays.
 */
struct GraphNumpy {
    NodesNumpy nodes;
    pybind11::array edges;
    pybind11::array record_offsets;
    pybind11::array assembly_nodes;
    pybind11::array node_offsets;
};

/**
 * @brief View a NumPy-backed graph without copying its arrays.
 *
 * The graph's arrays must remain valid and unchanged for the lifetime of the view.
 */
GraphView make_view(const GraphNumpy& graph);

/**
 * @brief Provide the Python-facing graph over native or NumPy-backed storage.
 *
 * Native algorithms access graph arrays through read-only views.
 */
class GraphPython {
public:
    explicit GraphPython(Graph&& graph);

    /** Return native storage, or nullptr for a NumPy-backed graph. */
    const Graph* native() const noexcept;

    /** Return NumPy-backed storage, or nullptr for a native graph. */
    const GraphNumpy* numpy() const noexcept;

    /**
     * Return a unified, non-owning graph view for downstream C++ processing,
     * independent of the underlying storage backend.
     */
    GraphView view() const;

    /**
     * Save a native graph as a directory of NumPy arrays (`.npy` files).
     * Existing files are overwritten.
     */
    void save(const pybind11::object& path) const;

    /**
     * Construct a graph backed by NumPy arrays without copying.
     * Validates array metadata and basic graph invariants.
     */
    static GraphPython from_numpy(GraphNumpy arrays);

    /**
     * Load a graph from NumPy files using read-only memory mapping.
     * Delegates to `from_numpy()` for array validation and graph construction.
     */
    static GraphPython load(const pybind11::object& path);

private:
    explicit GraphPython(GraphNumpy&& graph);
    std::variant<Graph, GraphNumpy> storage_;
};

void bind_python_graph(pybind11::module_& module);

void bind_python_classes(pybind11::module_& module);

} // namespace seqwin::bindings
