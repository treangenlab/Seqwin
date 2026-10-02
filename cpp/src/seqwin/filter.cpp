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

void calculate_thresholds(
    const Node* nodes,
    std::size_t n_nodes,
    const bool* is_targets,
    std::size_t n_assemblies,
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
        internal::log_python("Penalty threshold is provided (--penalty-th), skip auto estimation", "warning");
    } else {
        // Consider k-mers in target assemblies:
        double e_absence_tar; // their expected absence in target assemblies
        double e_presence_neg; // their expected presence in non-target assemblies
        if (jaccard) {
            if (jaccard_rows != n_assemblies || jaccard_cols != n_assemblies) {
                throw std::invalid_argument("Jaccard matrix shape must match the number of assemblies");
            }
            e_absence_tar = 1.0 - expected_presence_jaccard(jaccard, n_assemblies, is_targets, true);
            e_presence_neg = expected_presence_jaccard(jaccard, n_assemblies, is_targets, false);
        } else {
            std::tie(e_absence_tar, e_presence_neg) = internal::expected_presence(
                nodes,
                n_nodes,
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
            " - Subgraph size limit is set to [" + std::to_string(min_nodes) + ", " + std::to_string(*max_nodes) + "]"
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
    const Kmer* kmers,
    const Node* nodes,
    std::size_t n_nodes,
    const Edge* edges,
    std::size_t n_edges,
    const std::uint32_t* record_offsets,
    std::size_t n_record_offsets,
    const std::size_t* assembly_nodes,
    std::size_t n_assembly_nodes,
    const std::size_t* node_offsets,
    std::size_t n_node_offsets,
    const std::vector<std::string>& assembly_paths,
    const bool* is_targets,
    std::size_t n_assemblies,
    const double* jaccard,
    std::size_t jaccard_rows,
    std::size_t jaccard_cols,
    const FilterConfig& config
) {
    internal::ThreadPool pool(std::max<std::size_t>(1, config.n_cpu));

    FilteredGraph filtered;
    filtered.total_tar = std::count(is_targets, is_targets + n_assemblies, true);
    filtered.total_neg = n_assemblies - filtered.total_tar;
    if (filtered.total_tar == 0) {
        throw std::invalid_argument("is_targets must contain at least one target assembly");
    }
    if (filtered.total_neg == 0) {
        throw std::invalid_argument("is_targets must contain at least one non-target assembly");
    }

    internal::log_python(" - Counting target node occurrences...");
    auto target_counts = internal::count_target_nodes(
        n_nodes,
        assembly_nodes,
        n_assembly_nodes,
        node_offsets,
        n_node_offsets,
        is_targets,
        n_assemblies,
        filtered.total_tar,
        filtered.total_neg,
        pool
    );
    calculate_thresholds(
        nodes,
        n_nodes,
        is_targets,
        n_assemblies,
        jaccard,
        jaccard_rows,
        jaccard_cols,
        target_counts,
        config,
        filtered,
        pool
    );

    internal::log_python(" - Filtering graph and calculating node penalty scores...");
    internal::prune_graph(
        nodes,
        n_nodes,
        edges,
        n_edges,
        target_counts,
        filtered.total_tar,
        filtered.total_neg,
        filtered.edge_weight_th,
        filtered,
        pool
    );
    internal::log_python(
        " - Removed " + std::to_string(n_edges - filtered.edges.size()) + " edges with weight<" +
        format_value(filtered.edge_weight_th, 3) + ", " + std::to_string(filtered.edges.size()) + " edges left"
    );
    internal::log_python(
        " - Removed " + std::to_string(n_nodes - filtered.nodes.size()) + " isolated nodes, " +
        std::to_string(filtered.nodes.size()) + " nodes left"
    );

    internal::get_subgraphs(
        filtered.nodes,
        filtered.edges,
        filtered.penalty_th,
        filtered.min_nodes,
        filtered.max_nodes,
        filtered
    );
    if (filtered.subgraphs.empty()) {
        throw std::runtime_error("No low-penalty subgraph was found. Try decrease --stringency, or increase --penalty-th");
    }
    internal::log_python(" - Found " + std::to_string(filtered.subgraphs.size()) + " low-penalty subgraphs");

    internal::log_python(" - Finding a representative for each low-penalty subgraph...");
    auto signatures = internal::extract_signatures(
        filtered.subgraphs,
        filtered.nodes,
        kmers,
        nodes,
        n_nodes,
        record_offsets,
        n_record_offsets,
        assembly_paths,
        is_targets,
        n_assemblies,
        config.kmerlen,
        config.windowsize,
        config.min_len,
        config.consec_kmer_mul,
        filtered.total_tar,
        pool
    );

    return {std::move(filtered), std::move(signatures)};
}

} // namespace seqwin
