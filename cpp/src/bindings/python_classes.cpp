#include "bindings/python_classes.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "seqwin/filter.hpp"

namespace py = pybind11;

namespace {

// Make public classes appear as seqwin.core, not seqwin.core._native
constexpr char public_module[] = "seqwin.core";

// Define an explicit, versioned binary format
constexpr int pickle_version = 1;

static_assert(sizeof(std::size_t) == 8, "Seqwin requires a 64-bit platform");
static_assert(std::is_trivially_copyable_v<seqwin::Node>);
static_assert(std::is_standard_layout_v<seqwin::Node>);
static_assert(sizeof(seqwin::Node) == 40);
static_assert(offsetof(seqwin::Node, hash) == 0);
static_assert(offsetof(seqwin::Node, start) == 8);
static_assert(offsetof(seqwin::Node, stop) == 16);
static_assert(offsetof(seqwin::Node, n_tar) == 24);
static_assert(offsetof(seqwin::Node, n_neg) == 28);
static_assert(offsetof(seqwin::Node, penalty) == 32);
static_assert(std::is_trivially_copyable_v<seqwin::Edge>);
static_assert(std::is_standard_layout_v<seqwin::Edge>);
static_assert(sizeof(seqwin::Edge) == 24);
static_assert(offsetof(seqwin::Edge, first) == 0);
static_assert(offsetof(seqwin::Edge, second) == 8);
static_assert(offsetof(seqwin::Edge, weight) == 16);
static_assert(sizeof(double) == 8);
static_assert(std::numeric_limits<double>::is_iec559);

bool is_little_endian() {
    const std::uint16_t value = 1;
    return *reinterpret_cast<const std::uint8_t*>(&value) == 1;
}

template <typename T>
py::bytes array_to_bytes(const seqwin::NoInitArray<T>& array) {
    if (!is_little_endian()) {
        throw std::runtime_error("Seqwin pickle format requires a little-endian platform");
    }
    if (array.empty()) {
        return py::bytes();
    }
    return py::bytes(
        // py::bytes only accepts const char*
        reinterpret_cast<const char*>(array.data()),
        static_cast<py::ssize_t>(array.size() * sizeof(T))
    );
}

template <typename T>
seqwin::NoInitArray<T> bytes_to_array(const py::bytes& value, const char* name) {
    if (!is_little_endian()) {
        throw std::runtime_error("Seqwin pickle format requires a little-endian platform");
    }
    const std::string bytes = value;
    if (bytes.size() % sizeof(T) != 0) {
        throw std::runtime_error(
            std::string("Invalid FilteredGraph pickle: ") + name +
            " byte length is not a multiple of the record size"
        );
    }
    seqwin::NoInitArray<T> result(bytes.size() / sizeof(T));
    if (!bytes.empty()) {
        std::memcpy(result.data(), bytes.data(), bytes.size());
    }
    return result;
}

} // namespace

namespace seqwin::bindings {

void bind_python_classes(py::module_& module) {
    py::class_<FilteredGraph>(module, "FilteredGraph")
        .def_property_readonly("nodes", [](py::object self) {
            auto& filtered = self.cast<FilteredGraph&>();
            return py::array_t<Node>(
                {static_cast<py::ssize_t>(filtered.nodes.size())},
                {static_cast<py::ssize_t>(sizeof(Node))},
                filtered.nodes.data(),
                self
            );
        })
        .def_property_readonly("edges", [](py::object self) {
            auto& filtered = self.cast<FilteredGraph&>();
            return py::array_t<Edge>(
                {static_cast<py::ssize_t>(filtered.edges.size())},
                {static_cast<py::ssize_t>(sizeof(Edge))},
                filtered.edges.data(),
                self
            );
        })
        .def_readonly("subgraphs", &FilteredGraph::subgraphs)
        .def_readonly("total_tar", &FilteredGraph::total_tar)
        .def_readonly("total_neg", &FilteredGraph::total_neg)
        .def_readonly("e_absence_tar", &FilteredGraph::e_absence_tar)
        .def_readonly("e_presence_neg", &FilteredGraph::e_presence_neg)
        .def_readonly("penalty_th", &FilteredGraph::penalty_th)
        .def_readonly("edge_weight_th", &FilteredGraph::edge_weight_th)
        .def_readonly("min_nodes", &FilteredGraph::min_nodes)
        .def_readonly("max_nodes", &FilteredGraph::max_nodes)
        .def(py::pickle(
            [](const FilteredGraph& value) {
                return py::make_tuple(
                    pickle_version,
                    array_to_bytes(value.nodes),
                    array_to_bytes(value.edges),
                    value.subgraphs,
                    value.total_tar,
                    value.total_neg,
                    value.e_absence_tar,
                    value.e_presence_neg,
                    value.penalty_th,
                    value.edge_weight_th,
                    value.min_nodes,
                    value.max_nodes
                );
            },
            [](const py::tuple& state) {
                if (state.size() != 12 || state[0].cast<int>() != pickle_version) {
                    throw std::runtime_error("Invalid FilteredGraph pickle state or version");
                }
                return FilteredGraph{
                    bytes_to_array<Node>(state[1].cast<py::bytes>(), "nodes"),
                    bytes_to_array<Edge>(state[2].cast<py::bytes>(), "edges"),
                    state[3].cast<std::vector<Subgraph>>(),
                    state[4].cast<std::size_t>(),
                    state[5].cast<std::size_t>(),
                    state[6].cast<double>(),
                    state[7].cast<double>(),
                    state[8].cast<double>(),
                    state[9].cast<double>(),
                    state[10].cast<std::size_t>(),
                    state[11].cast<std::optional<std::size_t>>()
                };
            }
        ))
        .attr("__module__") = public_module;

    py::class_<SubgraphLoc>(module, "SubgraphLoc")
        .def_readonly("assembly_idx", &SubgraphLoc::assembly_idx)
        .def_readonly("record_idx", &SubgraphLoc::record_idx)
        .def_readonly("start", &SubgraphLoc::start)
        .def_readonly("stop", &SubgraphLoc::stop)
        .def_readonly("n_kmers", &SubgraphLoc::n_kmers)
        .def_readonly("n_repeats", &SubgraphLoc::n_repeats)
        .def(py::pickle(
            [](const SubgraphLoc& value) {
                return py::make_tuple(
                    pickle_version,
                    value.assembly_idx,
                    value.record_idx,
                    value.start,
                    value.stop,
                    value.n_kmers,
                    value.n_repeats
                );
            },
            [](const py::tuple& state) {
                if (state.size() != 7 || state[0].cast<int>() != pickle_version) {
                    throw std::runtime_error("Invalid SubgraphLoc pickle state or version");
                }
                return SubgraphLoc{
                    state[1].cast<std::size_t>(),
                    state[2].cast<std::size_t>(),
                    state[3].cast<std::size_t>(),
                    state[4].cast<std::size_t>(),
                    state[5].cast<std::size_t>(),
                    state[6].cast<std::size_t>()
                };
            }
        ))
        .attr("__module__") = public_module;

    py::class_<Signature>(module, "Signature")
        .def_readonly("subgraph_idx", &Signature::subgraph_idx)
        .def_readonly("location", &Signature::location)
        .def_readonly("sequence", &Signature::sequence)
        .def_readonly("length", &Signature::length)
        .def_readonly("n_rep", &Signature::n_rep)
        .def_readonly("rep_ratio", &Signature::rep_ratio)
        .def(py::pickle(
            [](const Signature& value) {
                return py::make_tuple(
                    pickle_version,
                    value.subgraph_idx,
                    value.location,
                    value.sequence,
                    value.length,
                    value.n_rep,
                    value.rep_ratio
                );
            },
            [](const py::tuple& state) {
                if (state.size() != 7 || state[0].cast<int>() != pickle_version) {
                    throw std::runtime_error("Invalid Signature pickle state or version");
                }
                return Signature{
                    state[1].cast<std::size_t>(),
                    state[2].cast<SubgraphLoc>(),
                    state[3].cast<std::string>(),
                    state[4].cast<std::size_t>(),
                    state[5].cast<std::size_t>(),
                    state[6].cast<double>()
                };
            }
        ))
        .attr("__module__") = public_module;
}

} // namespace seqwin::bindings
