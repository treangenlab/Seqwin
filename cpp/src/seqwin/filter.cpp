#include "seqwin/filter.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
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
double expected_presence(
    const double* jaccard,
    std::size_t n,
    const bool* is_targets,
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

/** Calculate thresholds and add them to `result`. */
void calculate_thresholds(
    const Node* nodes,
    std::size_t n_nodes,
    const bool* is_targets,
    std::size_t n_assemblies,
    const double* jaccard,
    std::size_t jaccard_rows,
    std::size_t jaccard_cols,
    const FilterConfig& config,
    FilterResult& result
) {
    const auto total_tar = result.total_tar;
    const auto total_neg = result.total_neg;

    double penalty_th;
    if (config.penalty_th) {
        penalty_th = *config.penalty_th;
        internal::log_python("Penalty threshold is provided (--penalty-th), skip auto estimation", "warning");
    } else {
        internal::log_python(" - Calculating penalty threshold...");
        // Consider k-mers in target assemblies:
        double e_absence_tar; // their expected absence in target assemblies
        double e_presence_neg; // their expected presence in non-target assemblies
        if (jaccard) {
            if (jaccard_rows != n_assemblies || jaccard_cols != n_assemblies) {
                throw std::invalid_argument("Jaccard matrix shape must match the number of assemblies");
            }
            e_absence_tar = 1.0 - expected_presence(jaccard, n_assemblies, is_targets, true);
            e_presence_neg = expected_presence(jaccard, n_assemblies, is_targets, false);
        } else {
            // Calculate expected presence from minimizer sketches
            // For all k-mers in targets, calculate their average presence in targets or non-targets
            double sum_n_tar = 0.0; // Number of k-mers in all targets
            double sum_presence_tar = 0.0;
            double sum_presence_neg = 0.0;
            for (std::size_t i = 0; i < n_nodes; ++i) {
                const double node_n_tar = nodes[i].n_tar;
                const double node_n_neg = nodes[i].n_neg;
                sum_n_tar += node_n_tar;
                sum_presence_tar += (node_n_tar / total_tar) * node_n_tar;
                sum_presence_neg += (node_n_neg / total_neg) * node_n_tar;
            }
            if (sum_n_tar == 0.0) {
                throw std::invalid_argument("No target minimizers are available for threshold estimation");
            }
            e_absence_tar = 1.0 - sum_presence_tar / sum_n_tar;
            e_presence_neg = sum_presence_neg / sum_n_tar;
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
    }

    // Calculate edge weight threshold
    // Consider N as the number of assemblies that include a certain k-mer. Since we want k-mers with
    // penalty lower than penalty_th, based on the definition of penalty, N ≥ (1 - penalty_th) * total_tar.
    // So edge weight threshold is calculated based on the lower bound of N, times a multiplier < 1.
    const double edge_weight_th = config.edge_w_th_mul * (1.0 - penalty_th) * total_tar;

    // Calculate size range of subgraphs
    const std::size_t gap_len = (config.windowsize + 1) / 2;
    const std::size_t min_nodes = std::max(config.min_nodes_floor, config.min_len / gap_len + 1);
    const std::optional<std::size_t> max_nodes = config.max_len
        ? std::optional<std::size_t>(*config.max_len / gap_len + 1)
        : config.max_nodes_cap;
    if (max_nodes) {
        internal::log_python(
            " - Subgraph size limit is set to [" + std::to_string(min_nodes) + ", " + std::to_string(*max_nodes) + "]"
        );
    } else {
        internal::log_python(
            " - Upper limit of subgraph size is not set. Lower limit is set to " + std::to_string(min_nodes),
            "warning"
        );
    }

    result.penalty_th = penalty_th;
    result.edge_weight_th = edge_weight_th;
    result.min_nodes = min_nodes;
    result.max_nodes = max_nodes;
}

} // namespace

FilterResult filter(
    const Kmer* kmers,
    Node* nodes,
    std::size_t n_nodes,
    const Edge* edges,
    std::size_t n_edges,
    const std::uint32_t* record_offsets,
    std::size_t n_record_offsets,
    const bool* is_targets,
    std::size_t n_assemblies,
    const double* jaccard,
    std::size_t jaccard_rows,
    std::size_t jaccard_cols,
    const FilterConfig& config
) {
    internal::log_python(" - Calculating node penalty scores...");
    auto result = internal::get_penalty(
        kmers, nodes, n_nodes, record_offsets, n_record_offsets, is_targets, n_assemblies, config.n_cpu
    );
    calculate_thresholds(
        nodes, n_nodes, is_targets, n_assemblies, jaccard, jaccard_rows, jaccard_cols, config, result
    );

    internal::log_python(" - Filtering graph edges and nodes...");
    auto pruned = internal::prune_graph(
        nodes, n_nodes, edges, n_edges, result.edge_weight_th
    );
    internal::log_python(
        " - Removed " + std::to_string(n_edges - pruned.edges.size()) + " edges with weight<" +
        format_value(result.edge_weight_th, 3) + ", " + std::to_string(pruned.edges.size()) + " edges left"
    );
    internal::log_python(
        " - Removed " + std::to_string(n_nodes - pruned.nodes.size()) + " isolated nodes, " +
        std::to_string(pruned.nodes.size()) + " nodes left"
    );

    auto [subgraphs, used_nodes] = internal::get_subgraphs(
        pruned.nodes, pruned.edges, result.penalty_th, result.min_nodes,
        result.max_nodes.value_or(std::numeric_limits<std::size_t>::max())
    );
    if (subgraphs.empty()) {
        throw std::runtime_error("No low-penalty subgraph was found. Try decrease --stringency, or increase --penalty-th");
    }
    internal::log_python(" - Found " + std::to_string(subgraphs.size()) + " low-penalty subgraphs");

    auto compacted = internal::compact_graph(
        kmers, pruned.nodes, pruned.edges, std::move(used_nodes)
    );
    internal::log_python(" - " + std::to_string(compacted.kmers.size()) + " k-mers left");

    result.kmers = std::move(compacted.kmers);
    result.nodes = std::move(compacted.nodes);
    result.edges = std::move(compacted.edges);
    result.subgraphs = std::move(subgraphs);
    return result;
}

} // namespace seqwin
