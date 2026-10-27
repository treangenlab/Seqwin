#pragma once

#include <variant>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "seqwin/graph_view.hpp"

namespace seqwin::bindings {

/** Python module path used by public classes. */
inline constexpr char public_module[] = "seqwin.core";

/**
 * @brief NumPy node arrays exposed through the Python graph API.
 *
 * For native graphs, they are read-only, zero-copy views of the native buffers.
 * For memory-mapped graphs, they are the original NumPy memmaps.
 */
struct NodesPy {
    pybind11::array kmers;
    pybind11::array offsets;
    pybind11::array prevalences;
    pybind11::array hashes;
};

/**
 * @brief Read-only memory-mapped NumPy arrays backing a graph.
 */
struct GraphMemmap {
    NodesPy nodes;
    pybind11::array edges;
    pybind11::array record_offsets;
    pybind11::array assembly_nodes;
    pybind11::array node_offsets;
};

/**
 * @brief View a memory-mapped graph without copying its arrays.
 *
 * The graph's memory maps must remain valid for the lifetime of the view.
 */
GraphView make_view(const GraphMemmap& graph);

/**
 * @brief Provide the Python-facing graph over native or memory-mapped storage.
 */
class GraphPy {
public:
    explicit GraphPy(Graph&& graph);
    explicit GraphPy(GraphMemmap&& graph);

    /** Return native storage, or nullptr for a memory-mapped graph. */
    const Graph* native() const noexcept;

    /** Return memory-mapped storage, or nullptr for a native graph. */
    const GraphMemmap* memmap() const noexcept;

    /** Return a non-owning graph view for downstream C++ processing. */
    GraphView view() const;

    /** Save native graph arrays as NumPy files. */
    void save(const pybind11::object& path) const;

    /** Load a graph backed by read-only NumPy memory maps. */
    static GraphPy load(const pybind11::object& path);

private:
    std::variant<Graph, GraphMemmap> storage_;
};

void bind_python_graph(pybind11::module_& module);

void bind_python_classes(pybind11::module_& module);

} // namespace seqwin::bindings
