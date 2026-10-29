#include "bindings/python_bindings.hpp"

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "seqwin/filter.hpp"
#include "seqwin/graph.hpp"
#include "bindings/python_classes.hpp"

namespace py = pybind11;

namespace seqwin::bindings {

std::vector<std::string> normalize_paths(const py::handle& assembly_paths) {
    if (
        py::isinstance<py::str>(assembly_paths) ||
        py::isinstance<py::bytes>(assembly_paths) ||
        !py::isinstance<py::sequence>(assembly_paths)
    ) {
        throw py::type_error("assembly_paths must be a sequence of str or os.PathLike[str] paths");
    }
    const auto sequence = py::reinterpret_borrow<py::sequence>(assembly_paths);
    std::vector<std::string> paths;
    paths.reserve(sequence.size());
    for (const auto& path : sequence) {
        auto normalized = py::reinterpret_steal<py::object>(PyOS_FSPath(path.ptr()));
        if (!normalized) {
            throw py::error_already_set();
        }
        paths.push_back(normalized.cast<std::string>());
    }
    return paths;
}

} // namespace seqwin::bindings

namespace {

/**
 * @brief A reusable collection of assembly paths for native C++ calls.
 *
 * Paths are normalized using `os.fspath()` and stored as C++ strings,
 * avoiding repeated conversions.
 */
class AssemblyPaths {
public:
    explicit AssemblyPaths(const py::sequence& paths)
        : paths_(seqwin::bindings::normalize_paths(paths))
    {}

    const std::vector<std::string>& paths() const noexcept {
        return paths_;
    }

private:
    std::vector<std::string> paths_;
};

} // namespace

using seqwin::bindings::NumpyArray;
using seqwin::bindings::validate_numpy;
using seqwin::bindings::view_numpy;

PYBIND11_MODULE(_native, m) {
    PYBIND11_NUMPY_DTYPE(seqwin::Kmer, pos, record_idx);
    PYBIND11_NUMPY_DTYPE(seqwin::FilteredNode, idx, n_tar, n_neg, penalty);
    PYBIND11_NUMPY_DTYPE(seqwin::Edge, first, second, weight);

    seqwin::bindings::bind_python_classes(m);

    py::class_<AssemblyPaths>(m, "AssemblyPaths")
        .def(py::init<const py::sequence&>(), py::arg("assembly_paths"))
        .def("__len__", [](const AssemblyPaths& paths) {
            return paths.paths().size();
        })
        .attr("__module__") = seqwin::bindings::public_module;

    py::tuple names(seqwin::METADATA_NAMES.size());
    for (std::size_t i = 0; i < seqwin::METADATA_NAMES.size(); ++i) {
        names[i] = py::str(std::string(seqwin::METADATA_NAMES[i]));
    }
    m.attr("_METADATA_NAMES") = std::move(names);

    m.def("_filter_native",
        [](const seqwin::bindings::GraphPython& graph,
           py::object assembly_paths,
           NumpyArray<bool> is_targets,
           std::optional<NumpyArray<double>> jaccard,
           std::size_t kmerlen,
           std::size_t windowsize,
           std::optional<double> penalty_th,
           double stringency,
           std::size_t min_len,
           std::optional<std::size_t> max_len,
           double penalty_th_cap,
           double edge_w_th_mul,
           std::size_t min_nodes_floor,
           std::optional<std::size_t> max_nodes_cap,
           double consec_kmer_mul,
           std::size_t n_cpu,
           const std::optional<std::string>& fasta_path,
           const std::optional<std::string>& csv_path
        ) {
            const auto graph_view = graph.view();

            std::vector<std::string> normalized_paths;
            const auto& paths = py::isinstance<AssemblyPaths>(assembly_paths)
                ? assembly_paths.cast<const AssemblyPaths&>().paths()
                : (normalized_paths = seqwin::bindings::normalize_paths(assembly_paths));

            validate_numpy<bool>(is_targets, "is_targets");
            const auto is_targets_view = view_numpy<bool>(is_targets);

            const double* jaccard_ptr = nullptr;
            std::size_t jaccard_rows = 0;
            std::size_t jaccard_cols = 0;
            if (jaccard) {
                if (jaccard->ndim() != 2) {
                    throw std::invalid_argument("jaccard must be a two-dimensional array");
                }
                if (reinterpret_cast<std::uintptr_t>(jaccard->data()) % alignof(double) != 0) {
                    throw std::invalid_argument("jaccard must be aligned");
                }
                jaccard_ptr = jaccard->data();
                jaccard_rows = static_cast<std::size_t>(jaccard->shape(0));
                jaccard_cols = static_cast<std::size_t>(jaccard->shape(1));
            }

            seqwin::FilterConfig config{
                kmerlen,
                windowsize,
                penalty_th,
                stringency,
                min_len,
                max_len,
                penalty_th_cap,
                edge_w_th_mul,
                min_nodes_floor,
                max_nodes_cap,
                consec_kmer_mul,
                n_cpu,
                fasta_path,
                csv_path
            };

            std::pair<seqwin::FilteredGraph, std::vector<seqwin::Signature>> out;
            {
                py::gil_scoped_release release;
                out = seqwin::filter(
                    graph_view,
                    paths,
                    is_targets_view,
                    jaccard_ptr,
                    jaccard_rows,
                    jaccard_cols,
                    config
                );
            }
            return py::make_tuple(std::move(out.first), std::move(out.second));
        },
        py::arg("graph"),
        py::arg("assembly_paths"),
        py::arg("is_targets").noconvert(),
        py::arg("jaccard").noconvert(),
        py::arg("kmerlen"),
        py::arg("windowsize"),
        py::arg("penalty_th"),
        py::arg("stringency"),
        py::arg("min_len"),
        py::arg("max_len"),
        py::arg("penalty_th_cap"),
        py::arg("edge_w_th_mul"),
        py::arg("min_nodes_floor"),
        py::arg("max_nodes_cap"),
        py::arg("consec_kmer_mul"),
        py::arg("n_cpu") = 1,
        py::arg("fasta_path") = py::none(),
        py::arg("csv_path") = py::none()
    );

}
