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

/**
 * @brief Expose a native C++ array as a read-only, zero-copy NumPy array.
 *
 * The NumPy array accesses the original C++ buffer directly and
 * keeps `owner` alive to prevent the buffer from being freed.
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
seqwin::ArrayView<T> numpy_to_view(const py::array& array) {
    return {
        static_cast<const T*>(array.data()),
        static_cast<std::size_t>(array.size())
    };
}

template <typename T>
py::array load_numpy(
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

/**
 * @brief Expose a native or memory-mapped array as a NumPy array.
 *
 * Return the original NumPy memmap for mapped storage,
 * or a read-only, zero-copy NumPy array for native storage.
 */
template <typename Container>
py::array expose_array(
    const py::object& owner,
    Container seqwin::Graph::* native_member,
    py::array seqwin::bindings::GraphMmap::* mapped_member
) {
    const auto& graph = owner.cast<const seqwin::bindings::GraphPy&>();
    if (const auto* mapped = graph.mmap()) {
        return mapped->*mapped_member;
    }
    return to_numpy(graph.native()->*native_member, owner);
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
            numpy_to_view<Kmer>(graph.nodes.kmers),
            numpy_to_view<std::size_t>(graph.nodes.offsets),
            numpy_to_view<std::uint32_t>(graph.nodes.prevalences),
            numpy_to_view<std::uint64_t>(graph.nodes.hashes)
        },
        numpy_to_view<Edge>(graph.edges),
        numpy_to_view<std::uint32_t>(graph.record_offsets),
        numpy_to_view<std::size_t>(graph.assembly_nodes),
        numpy_to_view<std::size_t>(graph.node_offsets)
    };
}

GraphPy::GraphPy(Graph&& graph)
    : storage_(std::move(graph))
{}

GraphPy::GraphPy(GraphMmap&& graph)
    : storage_(std::move(graph))
{}

const Graph* GraphPy::native() const noexcept {
    return std::get_if<Graph>(&storage_);
}

const GraphMmap* GraphPy::mmap() const noexcept {
    return std::get_if<GraphMmap>(&storage_);
}

GraphView GraphPy::view() const {
    return std::visit(
        [](const auto& graph) {
            return make_view(graph);
        },
        storage_
    );
}

void GraphPy::save(const py::object& path) const {
    if (mmap()) {
        throw std::invalid_argument("Cannot save a memory-mapped graph");
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

GraphPy GraphPy::load(const py::object& path) {
    const auto directory = py::module_::import("pathlib").attr("Path")(path);
    if (!directory.attr("is_dir")().cast<bool>()) {
        PyErr_SetString(
            PyExc_NotADirectoryError,
            ("Not a graph directory: " + py::str(directory).cast<std::string>()).c_str()
        );
        throw py::error_already_set();
    }
    for (const auto* name : graph_array_names) {
        const auto file = directory.attr("joinpath")(std::string(name) + ".npy");
        if (!file.attr("is_file")().cast<bool>()) {
            PyErr_SetString(
                PyExc_FileNotFoundError,
                (std::string("Missing graph array file: ") + name + ".npy").c_str()
            );
            throw py::error_already_set();
        }
    }
    GraphMmap graph{
        {
            load_numpy<Kmer>(directory, "kmers"),
            load_numpy<std::size_t>(directory, "offsets"),
            load_numpy<std::uint32_t>(directory, "prevalences"),
            load_numpy<std::uint64_t>(directory, "hashes")
        },
        load_numpy<Edge>(directory, "edges"),
        load_numpy<std::uint32_t>(directory, "record_offsets"),
        load_numpy<std::size_t>(directory, "assembly_nodes"),
        load_numpy<std::size_t>(directory, "node_offsets")
    };
    validate_graph(make_view(graph));
    return GraphPy(std::move(graph));
}

void bind_python_graph(py::module_& module) {
    py::class_<NodesPy>(module, "Nodes")
        .def_readonly("kmers", &NodesPy::kmers)
        .def_readonly("offsets", &NodesPy::offsets)
        .def_readonly("prevalences", &NodesPy::prevalences)
        .def_readonly("hashes", &NodesPy::hashes)
        .attr("__module__") = public_module;

    py::class_<GraphPy>(module, "Graph")
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
                return GraphPy(std::move(graph));
            }),
            py::arg("assembly_paths"),
            py::arg("kmerlen"),
            py::arg("windowsize"),
            py::arg("n_cpu") = 4,
            py::arg("low_memory") = false
        )
        .def_property_readonly("nodes", [](py::object self) {
            const auto& graph = self.cast<const GraphPy&>();
            if (const auto* mapped = graph.mmap()) {
                return mapped->nodes;
            }
            const auto& nodes = graph.native()->nodes;
            return NodesPy{
                to_numpy(nodes.kmers, self),
                to_numpy(nodes.offsets, self),
                to_numpy(nodes.prevalences, self),
                to_numpy(nodes.hashes, self)
            };
        })
        .def_property_readonly("edges", [](py::object self) {
            return expose_array(self, &Graph::edges, &GraphMmap::edges);
        })
        .def_property_readonly("record_offsets", [](py::object self) {
            return expose_array(self, &Graph::record_offsets, &GraphMmap::record_offsets);
        })
        .def_property_readonly("assembly_nodes", [](py::object self) {
            return expose_array(self, &Graph::assembly_nodes, &GraphMmap::assembly_nodes);
        })
        .def_property_readonly("node_offsets", [](py::object self) {
            return expose_array(self, &Graph::node_offsets, &GraphMmap::node_offsets);
        })
        .def("save", &GraphPy::save, py::arg("path"))
        .def_static("load", &GraphPy::load, py::arg("path"))
        .attr("__module__") = public_module;
}

} // namespace seqwin::bindings
