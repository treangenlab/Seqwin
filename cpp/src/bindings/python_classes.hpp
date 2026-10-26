#pragma once

#include <variant>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "seqwin/graph_view.hpp"

namespace seqwin::bindings {

/**
 * @brief Read-only NumPy memory-mapped graph arrays.
 */
struct GraphMmap {
    pybind11::array kmers;
    pybind11::array offsets;
    pybind11::array prevalences;
    pybind11::array hashes;
    pybind11::array edges;
    pybind11::array record_offsets;
    pybind11::array assembly_nodes;
    pybind11::array node_offsets;
};

GraphView make_view(const GraphMmap& graph);

/**
 * @brief Own either native graph buffers or NumPy memory maps.
 */
struct GraphHandle {
    explicit GraphHandle(Graph&& graph);
    explicit GraphHandle(GraphMmap&& graph);

    GraphView view() const;
    void save(const pybind11::object& path) const;
    static GraphHandle load(const pybind11::object& path);

    std::variant<Graph, GraphMmap> storage;
};

/**
 * @brief Keep the parent graph alive while its node arrays are accessible.
 */
struct NodesHandle {
    pybind11::object graph;
};

void bind_python_graph(pybind11::module_& module);

void bind_python_classes(pybind11::module_& module);

} // namespace seqwin::bindings
