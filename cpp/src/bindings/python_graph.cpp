#include "bindings/python_classes.hpp"

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

constexpr char public_module[] = "seqwin.core";

std::string path_string(const py::handle& path) {
    return py::module_::import("os").attr("fspath")(path).cast<std::string>();
}

template <typename T>
py::array readonly_array(
    seqwin::ArrayView<T> values,
    const py::handle& owner
) {
    // A non-null pointer also keeps the owner attached to empty NumPy views.
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

template <typename T>
seqwin::ArrayView<T> mapped_view(const py::array& array) {
    return {
        static_cast<const T*>(array.data()),
        static_cast<std::size_t>(array.size())
    };
}

template <typename T>
py::array load_array(
    const py::object& directory,
    const char* name
) {
    const auto path = directory.attr("joinpath")(std::string(name) + ".npy");
    py::array array = py::module_::import("numpy").attr("load")(
        path,
        py::arg("mmap_mode") = "r",
        py::arg("allow_pickle") = false
    ).template cast<py::array>();
    const std::string label = std::string("Graph array '") + name + "'";

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
    if (
        reinterpret_cast<std::uintptr_t>(array.data()) % alignof(T) != 0
    ) {
        throw std::invalid_argument(label + " must be aligned");
    }
    return array;
}

template <typename T>
void validate_offsets(
    seqwin::ArrayView<T> offsets,
    const char* name
) {
    const std::string label = std::string("Graph array '") + name + "'";
    if (offsets.empty()) {
        throw std::invalid_argument(label + " must not be empty");
    }
    if (offsets[0] != 0) {
        throw std::invalid_argument(label + " must start at zero");
    }
}

void validate_graph(const seqwin::GraphView& graph) {
    const auto& nodes = graph.nodes;
    const auto n_nodes = nodes.size();
    if (nodes.prevalences.size() != n_nodes) {
        throw std::invalid_argument("Graph arrays 'prevalences' and 'hashes' must have equal lengths");
    }
    if (nodes.offsets.size() != n_nodes + 1) {
        throw std::invalid_argument("Graph array 'offsets' length must equal n_nodes + 1");
    }
    validate_offsets(nodes.offsets, "offsets");
    if (nodes.offsets[n_nodes] != nodes.kmers.size()) {
        throw std::invalid_argument("Graph array 'offsets' must end at len(kmers)");
    }
    validate_offsets(graph.record_offsets, "record_offsets");
    validate_offsets(graph.node_offsets, "node_offsets");
    if (graph.node_offsets.size() != graph.record_offsets.size()) {
        throw std::invalid_argument("Graph arrays 'node_offsets' and 'record_offsets' must have equal lengths");
    }
    const auto n_assemblies = graph.node_offsets.size() - 1;
    if (graph.node_offsets[n_assemblies] != graph.assembly_nodes.size()) {
        throw std::invalid_argument("Graph array 'node_offsets' must end at len(assembly_nodes)");
    }
}

template <typename T>
py::array graph_array(
    const py::object& owner,
    seqwin::ArrayView<T> values,
    py::array seqwin::bindings::GraphMmap::* member
) {
    const auto& graph = owner.cast<const seqwin::bindings::GraphHandle&>();
    if (const auto* mapped = std::get_if<seqwin::bindings::GraphMmap>(&graph.storage)) {
        return mapped->*member;
    }
    return readonly_array(values, owner);
}

constexpr const char* graph_array_names[] = {
    "kmers", "offsets", "prevalences", "hashes",
    "edges", "record_offsets", "assembly_nodes", "node_offsets"
};

} // namespace

namespace seqwin::bindings {

GraphView make_view(const GraphMmap& graph) {
    return {
        {
            mapped_view<Kmer>(graph.kmers),
            mapped_view<std::size_t>(graph.offsets),
            mapped_view<std::uint32_t>(graph.prevalences),
            mapped_view<std::uint64_t>(graph.hashes)
        },
        mapped_view<Edge>(graph.edges),
        mapped_view<std::uint32_t>(graph.record_offsets),
        mapped_view<std::size_t>(graph.assembly_nodes),
        mapped_view<std::size_t>(graph.node_offsets)
    };
}

GraphHandle::GraphHandle(Graph&& graph)
    : storage(std::move(graph))
{}

GraphHandle::GraphHandle(GraphMmap&& graph)
    : storage(std::move(graph))
{}

GraphView GraphHandle::view() const {
    return std::visit(
        [](const auto& graph) {
            return make_view(graph);
        },
        storage
    );
}

void GraphHandle::save(const py::object& path) const {
    const auto directory = py::module_::import("pathlib").attr("Path")(path_string(path));
    // Check all destinations before writing anything, including symlinks and hard links
    // to any of this graph's mapped files.
    if (const auto* mapped = std::get_if<GraphMmap>(&storage)) {
        const py::array* arrays[] = {
            &mapped->kmers, &mapped->offsets, &mapped->prevalences, &mapped->hashes,
            &mapped->edges, &mapped->record_offsets, &mapped->assembly_nodes, &mapped->node_offsets
        };
        for (const auto* name : graph_array_names) {
            const auto destination = directory.attr("joinpath")(std::string(name) + ".npy");
            if (!destination.attr("exists")().cast<bool>()) {
                continue;
            }
            for (const auto* array : arrays) {
                if (destination.attr("samefile")(array->attr("filename")).cast<bool>()) {
                    throw std::invalid_argument("Cannot overwrite a memory-mapped graph input file");
                }
            }
        }
    }
    const auto graph = view();
    const auto owner = py::cast(this, py::return_value_policy::reference);
    const py::array arrays[] = {
        readonly_array(graph.nodes.kmers, owner),
        readonly_array(graph.nodes.offsets, owner),
        readonly_array(graph.nodes.prevalences, owner),
        readonly_array(graph.nodes.hashes, owner),
        readonly_array(graph.edges, owner),
        readonly_array(graph.record_offsets, owner),
        readonly_array(graph.assembly_nodes, owner),
        readonly_array(graph.node_offsets, owner)
    };
    for (std::size_t i = 0; i < std::size(graph_array_names); ++i) {
        py::module_::import("numpy").attr("save")(
            directory.attr("joinpath")(std::string(graph_array_names[i]) + ".npy"),
            arrays[i],
            py::arg("allow_pickle") = false
        );
    }
}

GraphHandle GraphHandle::load(const py::object& path) {
    const auto directory = py::module_::import("pathlib").attr("Path")(path_string(path));
    if (!directory.attr("is_dir")().cast<bool>()) {
        PyErr_SetString(PyExc_NotADirectoryError, ("Not a graph directory: " + path_string(path)).c_str());
        throw py::error_already_set();
    }
    for (const auto* name : graph_array_names) {
        const auto file = directory.attr("joinpath")(std::string(name) + ".npy");
        if (!file.attr("is_file")().cast<bool>()) {
            PyErr_SetString(PyExc_FileNotFoundError, (std::string("Missing graph array file: ") + name + ".npy").c_str());
            throw py::error_already_set();
        }
    }
    GraphMmap graph{
        load_array<Kmer>(directory, "kmers"),
        load_array<std::size_t>(directory, "offsets"),
        load_array<std::uint32_t>(directory, "prevalences"),
        load_array<std::uint64_t>(directory, "hashes"),
        load_array<Edge>(directory, "edges"),
        load_array<std::uint32_t>(directory, "record_offsets"),
        load_array<std::size_t>(directory, "assembly_nodes"),
        load_array<std::size_t>(directory, "node_offsets")
    };
    validate_graph(make_view(graph));
    return GraphHandle(std::move(graph));
}

void bind_python_graph(py::module_& module) {
    py::class_<NodesHandle>(module, "Nodes")
        .def_property_readonly("kmers", [](const NodesHandle& self) {
            const auto& graph = self.graph.cast<const GraphHandle&>();
            return graph_array(self.graph, graph.view().nodes.kmers, &GraphMmap::kmers);
        })
        .def_property_readonly("offsets", [](const NodesHandle& self) {
            const auto& graph = self.graph.cast<const GraphHandle&>();
            return graph_array(self.graph, graph.view().nodes.offsets, &GraphMmap::offsets);
        })
        .def_property_readonly("prevalences", [](const NodesHandle& self) {
            const auto& graph = self.graph.cast<const GraphHandle&>();
            return graph_array(self.graph, graph.view().nodes.prevalences, &GraphMmap::prevalences);
        })
        .def_property_readonly("hashes", [](const NodesHandle& self) {
            const auto& graph = self.graph.cast<const GraphHandle&>();
            return graph_array(self.graph, graph.view().nodes.hashes, &GraphMmap::hashes);
        })
        .attr("__module__") = public_module;

    py::class_<GraphHandle>(module, "Graph")
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
                    paths.push_back(path_string(path));
                }
                Graph graph;
                {
                    py::gil_scoped_release release;
                    graph = build(paths, kmerlen, windowsize, n_cpu, low_memory);
                }
                return GraphHandle(std::move(graph));
            }),
            py::arg("assembly_paths"),
            py::arg("kmerlen"),
            py::arg("windowsize"),
            py::arg("n_cpu") = 4,
            py::arg("low_memory") = false
        )
        .def_property_readonly("nodes", [](py::object self) {
            return NodesHandle{std::move(self)};
        })
        .def_property_readonly("edges", [](py::object self) {
            const auto& graph = self.cast<const GraphHandle&>();
            return graph_array(self, graph.view().edges, &GraphMmap::edges);
        })
        .def_property_readonly("record_offsets", [](py::object self) {
            const auto& graph = self.cast<const GraphHandle&>();
            return graph_array(self, graph.view().record_offsets, &GraphMmap::record_offsets);
        })
        .def_property_readonly("assembly_nodes", [](py::object self) {
            const auto& graph = self.cast<const GraphHandle&>();
            return graph_array(self, graph.view().assembly_nodes, &GraphMmap::assembly_nodes);
        })
        .def_property_readonly("node_offsets", [](py::object self) {
            const auto& graph = self.cast<const GraphHandle&>();
            return graph_array(self, graph.view().node_offsets, &GraphMmap::node_offsets);
        })
        .def("save", &GraphHandle::save, py::arg("path"))
        .def_static("load", &GraphHandle::load, py::arg("path"))
        .attr("__module__") = public_module;
}

} // namespace seqwin::bindings
