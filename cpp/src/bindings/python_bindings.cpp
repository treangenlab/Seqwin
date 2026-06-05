#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "seqwin/graph.hpp"
#include "seqwin/helpers.hpp"

namespace py = pybind11;

namespace {

template <typename Array>
auto array_to_numpy(Array&& values) {
    using Owner = typename std::decay<Array>::type;
    using T = typename std::remove_cv<
        typename std::remove_pointer<decltype(std::declval<Owner&>().data())>::type
    >::type;

    auto* owner = new Owner(std::forward<Array>(values));
    auto capsule = py::capsule(owner, [](void* ptr) {
        delete static_cast<Owner*>(ptr);
    });

    return py::array_t<T>(
        {static_cast<py::ssize_t>(owner->size())},
        {static_cast<py::ssize_t>(sizeof(T))},
        owner->data(),
        capsule
    );
}

}  // namespace

PYBIND11_MODULE(_core, m) {
    PYBIND11_NUMPY_DTYPE(seqwin::Kmer, pos, record_idx);
    PYBIND11_NUMPY_DTYPE(seqwin::Node, hash, start, stop, n_tar, n_neg, penalty);
    PYBIND11_NUMPY_DTYPE(seqwin::Edge, first, second, weight);

    m.doc() = "Seqwin minimizer graph bindings";

    m.def("_build_native",
        [](const std::vector<std::string>& assembly_paths,
           std::size_t kmerlen,
           std::size_t windowsize,
           const std::vector<bool>& is_targets,
           std::size_t n_cpu
        ) {
            seqwin::Graph graph;
            {
                py::gil_scoped_release release;
                graph = seqwin::build(
                    assembly_paths,
                    kmerlen,
                    windowsize,
                    is_targets,
                    n_cpu
                );
            }

            auto kmers = array_to_numpy(std::move(graph.kmers));
            auto nodes = array_to_numpy(std::move(graph.nodes));
            auto edges = array_to_numpy(std::move(graph.edges));
            auto record_offsets = array_to_numpy(std::move(graph.record_offsets));

            py::list ids_by_assembly;
            for (const auto& ids : graph.ids_by_assembly) {
                py::tuple ids_tuple(ids.size());
                for (std::size_t i = 0; i < ids.size(); ++i) {
                    ids_tuple[i] = ids[i];
                }
                ids_by_assembly.append(ids_tuple);
            }

            return py::make_tuple(kmers, nodes, edges, record_offsets, ids_by_assembly);
        },
        py::arg("assembly_paths"),
        py::arg("kmerlen"),
        py::arg("windowsize"),
        py::arg("is_targets"),
        py::arg("n_cpu") = 1
    );

    m.def("_filter_kmers_native",
        [](py::array_t<seqwin::Kmer, py::array::c_style> kmers,
           py::array_t<seqwin::Node, py::array::c_style> nodes,
           const std::vector<std::uint64_t>& used_hashes
        ) {
            auto kmers_buf = kmers.request();
            auto nodes_buf = nodes.request();

            const auto* kmers_ptr = static_cast<const seqwin::Kmer*>(kmers_buf.ptr);
            const auto* nodes_ptr = static_cast<const seqwin::Node*>(nodes_buf.ptr);
            const auto n_nodes = static_cast<std::size_t>(nodes_buf.shape[0]);

            seqwin::Graph graph;
            {
                py::gil_scoped_release release;
                graph = seqwin::filter_kmers(
                    kmers_ptr,
                    nodes_ptr,
                    n_nodes,
                    used_hashes
                );
            }

            auto kmers_new = array_to_numpy(std::move(graph.kmers));
            auto nodes_new = array_to_numpy(std::move(graph.nodes));

            return py::make_tuple(kmers_new, nodes_new);
        },
        py::arg("kmers").noconvert(),
        py::arg("nodes").noconvert(),
        py::arg("used_hashes")
    );
}
