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
 * For native graphs, arrays are read-only, zero-copy views that retain
 * the parent graph. For mapped graphs, they are the original NumPy
 * memmaps, which keep their mapped storage alive independently.
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
struct GraphMmap {
    NodesPy nodes;
    pybind11::array edges;
    pybind11::array record_offsets;
    pybind11::array assembly_nodes;
    pybind11::array node_offsets;
};

GraphView make_view(const GraphMmap& graph);

/**
 * @brief Provide the Python-facing graph over native or memory-mapped storage.
 */
class GraphPy {
public:
    explicit GraphPy(Graph&& graph);
    explicit GraphPy(GraphMmap&& graph);

    /** Return native storage, or nullptr for a memory-mapped graph. */
    const Graph* native() const noexcept;

    /** Return memory-mapped storage, or nullptr for a native graph. */
    const GraphMmap* mmap() const noexcept;

    /** Return a non-owning graph view for downstream C++ processing. */
    GraphView view() const;

    /** Save native graph arrays as NumPy files. */
    void save(const pybind11::object& path) const;

    /** Load a graph backed by read-only NumPy memory maps. */
    static GraphPy load(const pybind11::object& path);

private:
    std::variant<Graph, GraphMmap> storage_;
};

void bind_python_graph(pybind11::module_& module);

void bind_python_classes(pybind11::module_& module);

} // namespace seqwin::bindings
