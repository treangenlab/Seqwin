#include "bindings/python_bindings.hpp"

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "bindings/python_classes.hpp"
#include "seqwin/filter.hpp"
#include "seqwin/graph.hpp"
#include "utils/fasta_reader.hpp"

namespace py = pybind11;

namespace seqwin::bindings {
namespace {

std::string normalize_path(const py::handle& path) {
    auto normalized = py::reinterpret_steal<py::object>(PyOS_FSPath(path.ptr()));
    if (!normalized) {
        throw py::error_already_set();
    }
    return normalized.cast<std::string>();
}

/**
 * @brief A reusable collection of assembly paths for native C++ calls.
 *
 * Paths are normalized using `os.fspath()` and stored as C++ strings,
 * avoiding repeated conversions.
 */
class AssemblyPaths {
public:
    explicit AssemblyPaths(const py::sequence& paths)
        : paths_(normalize_paths(paths))
    {}

    const std::vector<std::string>& paths() const noexcept {
        return paths_;
    }

private:
    std::vector<std::string> paths_;
};

} // namespace

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
        paths.push_back(normalize_path(path));
    }
    return paths;
}

} // namespace seqwin::bindings

using namespace seqwin::bindings;

PYBIND11_MODULE(_native, m) {
    PYBIND11_NUMPY_DTYPE(seqwin::Kmer, pos, record_idx);
    PYBIND11_NUMPY_DTYPE(seqwin::FilteredNode, idx, n_tar, n_neg);
    PYBIND11_NUMPY_DTYPE(seqwin::Edge, first, second, weight);

    py::tuple names(seqwin::METADATA_NAMES.size());
    for (std::size_t i = 0; i < seqwin::METADATA_NAMES.size(); ++i) {
        names[i] = py::str(std::string(seqwin::METADATA_NAMES[i]));
    }
    m.attr("_METADATA_NAMES") = std::move(names);

    py::class_<AssemblyPaths>(m, "AssemblyPaths")
        .def(py::init<const py::sequence&>(), py::arg("assembly_paths"))
        .def("__len__", [](const AssemblyPaths& paths) {
            return paths.paths().size();
        })
        .attr("__module__") = public_module;

    py::class_<seqwin::internal::FastaRecord>(m, "FastaRecord")
        .def_readonly("id", &seqwin::internal::FastaRecord::id)
        .def_readonly("sequence", &seqwin::internal::FastaRecord::sequence)
        .attr("__module__") = public_module;

    bind_python_classes(m);

    m.def("read_fasta",
        [](const py::object& path) {
            const auto assembly_path = normalize_path(path);

            std::vector<seqwin::internal::FastaRecord> records;
            {
                py::gil_scoped_release release;
                records = seqwin::internal::read_fasta(assembly_path);
            }
            return records;
        },
        py::arg("path")
    );
    m.attr("read_fasta").attr("__module__") = public_module;

    m.def("_filter_native",
        [](const GraphPython& graph,
           py::object assembly_paths,
           NumpyArray<bool> is_targets,
           std::size_t kmerlen,
           std::size_t windowsize,
           double tar_weight,
           double neg_rank,
           std::size_t min_len,
           std::optional<std::size_t> max_len,
           double edge_w_th_mul,
           std::optional<std::size_t> n_tar_th,
           std::optional<std::size_t> n_neg_th,
           std::size_t min_nodes_floor,
           std::optional<std::size_t> max_nodes_cap,
           double consec_kmer_mul,
           std::size_t n_cpu,
           const std::optional<std::string>& fasta_path,
           const std::optional<std::string>& csv_path
        ) {
            const auto graph_view = graph.view();

            std::vector<std::string> normalized;
            const auto& paths = py::isinstance<AssemblyPaths>(assembly_paths)
                ? assembly_paths.cast<const AssemblyPaths&>().paths()
                : (normalized = normalize_paths(assembly_paths));

            validate_numpy<bool>(is_targets, "is_targets");
            const auto is_targets_view = view_numpy<bool>(is_targets);

            seqwin::FilterConfig config{
                kmerlen,
                windowsize,
                tar_weight,
                neg_rank,
                min_len,
                max_len,
                edge_w_th_mul,
                n_tar_th,
                n_neg_th,
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
                    config
                );
            }
            return py::make_tuple(std::move(out.first), std::move(out.second));
        },
        py::arg("graph"),
        py::arg("assembly_paths"),
        py::arg("is_targets").noconvert(),
        py::arg("kmerlen"),
        py::arg("windowsize"),
        py::arg("tar_weight"),
        py::arg("neg_rank"),
        py::arg("min_len"),
        py::arg("max_len"),
        py::arg("edge_w_th_mul"),
        py::arg("n_tar_th"),
        py::arg("n_neg_th"),
        py::arg("min_nodes_floor"),
        py::arg("max_nodes_cap"),
        py::arg("consec_kmer_mul"),
        py::arg("n_cpu") = 1,
        py::arg("fasta_path") = py::none(),
        py::arg("csv_path") = py::none()
    );

}
