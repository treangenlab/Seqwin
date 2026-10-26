#include "seqwin/filter.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "seqwin/filter_internals.hpp"
#include "utils/logging.hpp"

namespace seqwin {
namespace {

std::string format_value(double value, int precision)
{
    std::ostringstream out;
    out.precision(precision);
    out << std::fixed << value;
    return out.str();
}

/**
 * @brief Calculate the expected k-mer presence from pairwise Jaccard indices.
 *
 * Definition of presence `f(h)`: for a k-mer `h` in a group of `N` genomes (k-mer sets),
 * the fraction of genomes in a second group of `M` genomes that also contain `h`.
 *
 * Suppose `J` is the pairwise Jaccard matrix between genomes in the two groups, with shape `(M, N)`.
 * Then the expected presence can be calculated as: `E[f(h)] = mean(2J / (1+J))`.
 *
 * Here, the input matrix `jaccard` contains both target and non-target assemblies, with shape
 * `(M+N, M+N)`. The first group is always the target assemblies, and the second group is either
 * targets or non-targets. So the calculation only happens for certain rows and columns in `jaccard`.
 */
double expected_presence_jaccard(
    const double* jaccard,
    std::size_t n,
    ArrayView<bool> is_targets,
    bool vs_targets // If True, compare against target assemblies
) {
    double sum = 0.0;
    std::size_t count = 0;
    for (std::size_t row = 0; row < n; ++row) {
        if (!is_targets[row]) {
            // Always select targets for the first group
            continue;
        }
        for (std::size_t col = 0; col < n; ++col) {
            if (is_targets[col] != vs_targets) {
                // Select targets or non-targets for the second group
                continue;
            }
            const double value = jaccard[row * n + col];
            if (!std::isfinite(value) || value < 0.0 || value > 1.0) {
                throw std::invalid_argument("Jaccard values must be finite and between 0 and 1");
            }
            sum += 2.0 * value / (1.0 + value);
            ++count;
        }
    }
    if (count == 0) {
        throw std::invalid_argument("Jaccard matrix must not be empty");
    }
    return sum / static_cast<double>(count);
}

void calculate_thresholds(
    ArrayView<std::uint32_t> prevalences,
    ArrayView<bool> is_targets,
    const double* jaccard,
    std::size_t jaccard_rows,
    std::size_t jaccard_cols,
    const internal::TargetCounts& target_counts,
    const FilterConfig& config,
    FilteredGraph& filtered,
    internal::ThreadPool& pool
) {
    double penalty_th;
    if (config.penalty_th) {
        penalty_th = *config.penalty_th;
        internal::log_python(
            "Penalty threshold is provided (--penalty-th), skip auto estimation", "warning"
        );
    } else {
        // Consider k-mers in target assemblies:
        double e_absence_tar; // their expected absence in target assemblies
        double e_presence_neg; // their expected presence in non-target assemblies
        if (jaccard) {
            if (jaccard_rows != is_targets.size() || jaccard_cols != is_targets.size()) {
                throw std::invalid_argument("Jaccard matrix shape must match the number of assemblies");
            }
            e_absence_tar = 1.0 - expected_presence_jaccard(jaccard, is_targets.size(), is_targets, true);
            e_presence_neg = expected_presence_jaccard(jaccard, is_targets.size(), is_targets, false);
        } else {
            std::tie(e_absence_tar, e_presence_neg) = internal::expected_presence(
                prevalences,
                target_counts,
                filtered.total_tar,
                filtered.total_neg,
                pool
            );
        }
        internal::log_python(" - Expected k-mer absence in targets: " + format_value(e_absence_tar, 5));
        internal::log_python(" - Expected k-mer presence in non-targets: " + format_value(e_presence_neg, 5));
        penalty_th = (1.0 - config.stringency / 10.0) * std::sqrt(e_absence_tar * e_presence_neg);
        internal::log_python(" - Calculated penalty threshold: " + format_value(penalty_th, 5));

        if (penalty_th > config.penalty_th_cap) {
            penalty_th = config.penalty_th_cap;
            internal::log_python(
                " - Calculated penalty threshold is too large (capped at " + format_value(penalty_th, 5) + ")",
                "warning"
            );
        }
        filtered.e_absence_tar = e_absence_tar;
        filtered.e_presence_neg = e_presence_neg;
    }

    // Calculate edge weight threshold
    // Consider N as the number of assemblies that include a certain k-mer. Since we want k-mers with
    // penalty lower than penalty_th, based on the definition of penalty, N ≥ (1 - penalty_th) * total_tar.
    // So edge weight threshold is calculated based on the lower bound of N, times a multiplier < 1.
    const double edge_weight_th = config.edge_w_th_mul * (1.0 - penalty_th) * filtered.total_tar;

    // Calculate size range of subgraphs
    const std::size_t gap_len = (config.windowsize + 1) / 2;
    const std::size_t min_nodes = std::max(config.min_nodes_floor, config.min_len / gap_len + 1);
    const std::optional<std::size_t> max_nodes = config.max_len
        ? std::optional<std::size_t>(*config.max_len / gap_len + 1)
        : config.max_nodes_cap;
    if (max_nodes) {
        internal::log_python(
            " - Subgraph size limit is set to [" +
            std::to_string(min_nodes) + ", " + std::to_string(*max_nodes) + "]"
        );
    } else {
        internal::log_python(
            " - Upper limit of subgraph size is not set. Lower limit is set to " + std::to_string(min_nodes),
            "warning"
        );
    }

    filtered.penalty_th = penalty_th;
    filtered.edge_weight_th = edge_weight_th;
    filtered.min_nodes = min_nodes;
    filtered.max_nodes = max_nodes;
}

} // namespace

std::pair<FilteredGraph, std::vector<Signature>> filter(
    const GraphView& graph,
    const std::vector<std::string>& assembly_paths,
    ArrayView<bool> is_targets,
    const double* jaccard,
    std::size_t jaccard_rows,
    std::size_t jaccard_cols,
    const FilterConfig& config
) {
    if (config.fasta_path.has_value() != config.csv_path.has_value()) {
        throw std::invalid_argument("Both fasta_path and csv_path are required for signature output");
    }
    if (
        graph.nodes.offsets.size() == 0 ||
        graph.nodes.offsets.size() - 1 != graph.nodes.prevalences.size()
    ) {
        throw std::invalid_argument("offsets length must equal n_nodes + 1");
    }
    if (
        graph.nodes.offsets[0] != 0 ||
        graph.nodes.offsets[graph.nodes.prevalences.size()] != graph.nodes.kmers.size()
    ) {
        throw std::invalid_argument("offsets must start at zero and end at n_kmers");
    }
    internal::ThreadPool pool(std::max<std::size_t>(1, config.n_cpu));

    FilteredGraph filtered;
    filtered.total_tar = std::count(is_targets.begin(), is_targets.end(), true);
    filtered.total_neg = is_targets.size() - filtered.total_tar;
    if (filtered.total_tar == 0) {
        throw std::invalid_argument("is_targets must contain at least one target assembly");
    }
    if (filtered.total_neg == 0) {
        throw std::invalid_argument("is_targets must contain at least one non-target assembly");
    }

    internal::log_python(" - Counting target node occurrences...");
    auto target_counts = internal::count_target_nodes(
        graph.nodes.prevalences,
        graph.assembly_nodes,
        graph.node_offsets,
        is_targets,
        filtered.total_tar,
        filtered.total_neg,
        pool
    );
    calculate_thresholds(
        graph.nodes.prevalences,
        is_targets,
        jaccard,
        jaccard_rows,
        jaccard_cols,
        target_counts,
        config,
        filtered,
        pool
    );

    internal::log_python(" - Filtering graph and calculating node penalty scores...");
    auto graph_topology = internal::prune_graph(
        graph.nodes.prevalences,
        graph.edges,
        target_counts,
        filtered.total_tar,
        filtered.total_neg,
        filtered.edge_weight_th,
        filtered,
        pool
    );
    internal::log_python(
        " - Removed " + std::to_string(graph.edges.size() - filtered.edges.size()) +
        " edges failing weight/target-support thresholds, " +
        std::to_string(filtered.edges.size()) + " edges left"
    );
    internal::log_python(
        " - Removed " + std::to_string(graph.nodes.prevalences.size() - filtered.nodes.size()) +
        " isolated nodes, " + std::to_string(filtered.nodes.size()) + " nodes left"
    );

    internal::get_subgraphs(
        graph_topology,
        filtered.nodes,
        filtered.penalty_th,
        filtered.min_nodes,
        filtered.max_nodes,
        filtered
    );
    if (filtered.subgraphs.empty()) {
        throw std::runtime_error(
            "No low-penalty subgraph was found. Try decrease --stringency, or increase --penalty-th"
        );
    }
    internal::log_python(" - Found " + std::to_string(filtered.subgraphs.size()) + " low-penalty subgraphs");

    internal::log_python(" - Finding a representative for each low-penalty subgraph...");
    auto signatures = internal::extract_signatures(
        filtered.subgraphs,
        graph.nodes.kmers,
        graph.nodes.offsets,
        graph.record_offsets,
        assembly_paths,
        is_targets,
        config.kmerlen,
        config.windowsize,
        config.min_len,
        config.consec_kmer_mul,
        config.fasta_path,
        config.csv_path,
        filtered.total_tar,
        pool
    );

    return {std::move(filtered), std::move(signatures)};
}

} // namespace seqwin
