#include "bindings/python_classes.hpp"
#include "bindings/python_bindings.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "seqwin/build.hpp"

namespace py = pybind11;

namespace {

using seqwin::bindings::to_numpy;

template <typename T>
void validate_offsets(
    const py::array& offsets,
    const char* name
) {
    const std::string label = std::string("Graph array '") + name + "'";
    if (offsets.size() == 0) {
        throw std::invalid_argument(label + " must not be empty");
    }
    if (offsets.unchecked<T, 1>()(0) != 0) {
        throw std::invalid_argument(label + " must start at zero");
    }
}

void validate_graph(const seqwin::bindings::GraphNumpy& graph) {
    const auto& nodes = graph.nodes;
    const auto n_nodes = nodes.size();
    if (nodes.prevalences.size() != n_nodes) {
        throw std::invalid_argument("Graph arrays 'prevalences' and 'hashes' must have equal lengths");
    }
    if (nodes.offsets.size() != n_nodes + 1) {
        throw std::invalid_argument("Graph array 'offsets' length must equal n_nodes + 1");
    }
    validate_offsets<std::size_t>(nodes.offsets, "offsets");
    if (nodes.offsets.unchecked<std::size_t, 1>()(n_nodes) != nodes.kmers.size()) {
        throw std::invalid_argument("Graph array 'offsets' must end at len(kmers)");
    }
    validate_offsets<std::uint32_t>(graph.record_offsets, "record_offsets");
    validate_offsets<std::size_t>(graph.node_offsets, "node_offsets");
    if (graph.node_offsets.size() != graph.record_offsets.size()) {
        throw std::invalid_argument("Graph arrays 'node_offsets' and 'record_offsets' must have equal lengths");
    }
    const auto n_assemblies = graph.node_offsets.size() - 1;
    if (graph.node_offsets.unchecked<std::size_t, 1>()(n_assemblies) != graph.assembly_nodes.size()) {
        throw std::invalid_argument("Graph array 'node_offsets' must end at len(assembly_nodes)");
    }
}

/**
 * @brief Expose a native or NumPy-backed array to Python.
 *
 * Return a NumPy view for native storage, or the original NumPy array for NumPy-backed storage.
 */
template <typename Container>
py::array to_python(
    const py::object& owner,
    Container seqwin::Graph::* native_member,
    py::array seqwin::bindings::GraphNumpy::* numpy_member
) {
    const auto& graph = owner.cast<const seqwin::bindings::GraphPython&>();
    if (const auto* numpy = graph.numpy()) {
        return numpy->*numpy_member;
    }
    return to_numpy(graph.native()->*native_member, owner);
}

constexpr const char* graph_array_names[] = {
    "kmers", "offsets", "prevalences", "hashes",
    "edges", "record_offsets", "assembly_nodes", "node_offsets"
};

} // namespace

namespace seqwin::bindings {

GraphView make_view(const GraphNumpy& graph) {
    return {
        {
            view_numpy<Kmer>(graph.nodes.kmers),
            view_numpy<std::size_t>(graph.nodes.offsets),
            view_numpy<std::uint32_t>(graph.nodes.prevalences),
            view_numpy<std::uint64_t>(graph.nodes.hashes)
        },
        view_numpy<Edge>(graph.edges),
        view_numpy<std::uint32_t>(graph.record_offsets),
        view_numpy<std::size_t>(graph.assembly_nodes),
        view_numpy<std::size_t>(graph.node_offsets)
    };
}

GraphPython::GraphPython(Graph&& graph)
    : storage_(std::move(graph))
{}

GraphPython::GraphPython(GraphNumpy&& graph)
    : storage_(std::move(graph))
{}

const Graph* GraphPython::native() const noexcept {
    return std::get_if<Graph>(&storage_);
}

const GraphNumpy* GraphPython::numpy() const noexcept {
    return std::get_if<GraphNumpy>(&storage_);
}

GraphView GraphPython::view() const {
    return std::visit(
        [](const auto& graph) {
            return make_view(graph);
        },
        storage_
    );
}

void GraphPython::save(const py::object& path) const {
    if (numpy()) {
        throw std::invalid_argument("Cannot save a NumPy-backed graph");
    }
    const auto directory = py::module_::import("pathlib").attr("Path")(path);
    const auto& graph = *native();
    const auto owner = py::cast(this, py::return_value_policy::reference);
    const py::array arrays[] = {
        to_numpy(graph.nodes.kmers, owner),
        to_numpy(graph.nodes.offsets, owner),
        to_numpy(graph.nodes.prevalences, owner),
        to_numpy(graph.nodes.hashes, owner),
        to_numpy(graph.edges, owner),
        to_numpy(graph.record_offsets, owner),
        to_numpy(graph.assembly_nodes, owner),
        to_numpy(graph.node_offsets, owner)
    };
    const auto save = py::module_::import("numpy").attr("save");
    for (std::size_t i = 0; i < std::size(graph_array_names); ++i) {
        save(
            directory.attr("joinpath")(std::string(graph_array_names[i]) + ".npy"),
            arrays[i],
            py::arg("allow_pickle") = false
        );
    }
}

GraphPython GraphPython::from_numpy(GraphNumpy graph) {
    validate_numpy<Kmer>(graph.nodes.kmers, "kmers");
    validate_numpy<std::size_t>(graph.nodes.offsets, "offsets");
    validate_numpy<std::uint32_t>(graph.nodes.prevalences, "prevalences");
    validate_numpy<std::uint64_t>(graph.nodes.hashes, "hashes");
    validate_numpy<Edge>(graph.edges, "edges");
    validate_numpy<std::uint32_t>(graph.record_offsets, "record_offsets");
    validate_numpy<std::size_t>(graph.assembly_nodes, "assembly_nodes");
    validate_numpy<std::size_t>(graph.node_offsets, "node_offsets");

    validate_graph(graph);
    return GraphPython{std::move(graph)};
}

GraphPython GraphPython::load(const py::object& path) {
    const auto directory = py::module_::import("pathlib").attr("Path")(path);
    const auto numpy_load = py::module_::import("numpy").attr("load");
    const auto load = [&](const char* name) {
        return numpy_load(
            directory.attr("joinpath")(std::string(name) + ".npy"),
            py::arg("mmap_mode") = "r",
            py::arg("allow_pickle") = false
        ).cast<py::array>();
    };
    return GraphPython::from_numpy({
        {
            load("kmers"),
            load("offsets"),
            load("prevalences"),
            load("hashes")
        },
        load("edges"),
        load("record_offsets"),
        load("assembly_nodes"),
        load("node_offsets")
    });
}

void bind_python_graph(py::module_& module) {
    py::class_<NodesNumpy>(module, "Nodes")
        .def("__len__", &NodesNumpy::size)
        .def_readonly("kmers", &NodesNumpy::kmers)
        .def_readonly("offsets", &NodesNumpy::offsets)
        .def_readonly("prevalences", &NodesNumpy::prevalences)
        .def_readonly("hashes", &NodesNumpy::hashes)
        .attr("__module__") = public_module;

    py::class_<GraphPython>(module, "Graph")
        .def(py::init(
            [](const py::sequence& assembly_paths,
               std::size_t kmerlen,
               std::size_t windowsize,
               std::size_t n_cpu,
               bool low_memory
            ) {
                std::vector<std::string> paths;
                paths.reserve(assembly_paths.size());
                for (const auto& path : assembly_paths) {
                    paths.push_back(py::str(path).cast<std::string>());
                }
                Graph graph;
                {
                    py::gil_scoped_release release;
                    graph = build(paths, kmerlen, windowsize, n_cpu, low_memory);
                }
                return GraphPython(std::move(graph));
            }),
            py::arg("assembly_paths"),
            py::arg("kmerlen"),
            py::arg("windowsize"),
            py::arg("n_cpu") = 4,
            py::arg("low_memory") = false
        )
        .def_property_readonly("nodes", [](py::object self) {
            const auto& graph = self.cast<const GraphPython&>();
            if (const auto* numpy = graph.numpy()) {
                return numpy->nodes;
            }
            const auto& nodes = graph.native()->nodes;
            return NodesNumpy{
                to_numpy(nodes.kmers, self),
                to_numpy(nodes.offsets, self),
                to_numpy(nodes.prevalences, self),
                to_numpy(nodes.hashes, self)
            };
        })
        .def_property_readonly("edges", [](py::object self) {
            return to_python(self, &Graph::edges, &GraphNumpy::edges);
        })
        .def_property_readonly("record_offsets", [](py::object self) {
            return to_python(self, &Graph::record_offsets, &GraphNumpy::record_offsets);
        })
        .def_property_readonly("assembly_nodes", [](py::object self) {
            return to_python(self, &Graph::assembly_nodes, &GraphNumpy::assembly_nodes);
        })
        .def_property_readonly("node_offsets", [](py::object self) {
            return to_python(self, &Graph::node_offsets, &GraphNumpy::node_offsets);
        })
        .def("save", &GraphPython::save, py::arg("path"))
        .def_static("_from_numpy",
            [](NumpyArray<Kmer> kmers,
               NumpyArray<std::size_t> offsets,
               NumpyArray<std::uint32_t> prevalences,
               NumpyArray<std::uint64_t> hashes,
               NumpyArray<Edge> edges,
               NumpyArray<std::uint32_t> record_offsets,
               NumpyArray<std::size_t> assembly_nodes,
               NumpyArray<std::size_t> node_offsets
            ) {
                return GraphPython::from_numpy({
                    {
                        std::move(kmers),
                        std::move(offsets),
                        std::move(prevalences),
                        std::move(hashes)
                    },
                    std::move(edges),
                    std::move(record_offsets),
                    std::move(assembly_nodes),
                    std::move(node_offsets)
                });
            },
            py::kw_only(),
            py::arg("kmers").noconvert(),
            py::arg("offsets").noconvert(),
            py::arg("prevalences").noconvert(),
            py::arg("hashes").noconvert(),
            py::arg("edges").noconvert(),
            py::arg("record_offsets").noconvert(),
            py::arg("assembly_nodes").noconvert(),
            py::arg("node_offsets").noconvert()
        )
        .def_static("load", &GraphPython::load, py::arg("path"))
        .attr("__module__") = public_module;
}

} // namespace seqwin::bindings
