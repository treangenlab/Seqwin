#include "seqwin/filter.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "seqwin/filter_internals.hpp"
#include "utils/logging.hpp"

namespace seqwin {
namespace {

/**
 * @brief Calculate filtering thresholds from `config` and store them in `filtered`.
 */
void calculate_thresholds(
    std::size_t total_tar,
    const FilterConfig& config,
    FilteredGraph& filtered
) {
    // Calculate edge weight threshold
    const double edge_weight_th = config.edge_w_th_mul * total_tar;

    // Calculate size range of subgraphs
    const std::size_t gap_len = config.windowsize / 2 + config.windowsize % 2;
    const std::size_t min_nodes = std::max(config.min_nodes_floor, config.min_len / gap_len + 1);
    const std::optional<std::size_t> max_nodes = config.max_len
        ? std::optional<std::size_t>(*config.max_len / gap_len + 1)
        : config.max_nodes_cap;
    if (max_nodes) {
        if (*max_nodes < min_nodes) {
            throw std::invalid_argument(
                "min_nodes exceeds max_nodes. Try decreasing min_len or increasing max_len or max_nodes_cap"
            );
        }
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

    filtered.edge_weight_th = edge_weight_th;
    filtered.min_nodes = min_nodes;
    filtered.max_nodes = max_nodes;
}

/**
 * @brief Determine the non-target prevalence threshold.
 *
 * If `n_neg_th` is not provided, use the nearest-rank quantile (`neg_rank`) of non-target
 * prevalence among retained nodes whose target prevalence is at least `n_tar_th`.
 */
std::size_t neg_threshold(
    const NoInitArray<FilteredNode>& nodes,
    std::size_t n_tar_th,
    double neg_rank,
    std::optional<std::size_t> n_neg_th
) {
    if (n_neg_th) {
        return *n_neg_th;
    }
    std::vector<std::uint32_t> counts;
    for (const auto& node : nodes) {
        if (node.n_tar >= n_tar_th) {
            counts.push_back(node.n_neg);
        }
    }
    if (counts.empty()) {
        return 0;
    }
    const std::size_t index = std::max<std::size_t>(
        1,
        static_cast<std::size_t>(std::ceil(neg_rank * counts.size()))
    ) - 1;
    std::nth_element(counts.begin(), counts.begin() + index, counts.end());
    return counts[index];
}

} // namespace

std::pair<FilteredGraph, std::vector<Signature>> filter(
    const GraphView& graph,
    const std::vector<std::string>& assembly_paths,
    ArrayView<bool> is_targets,
    const FilterConfig& config
) {
    const std::size_t n_assemblies = graph.node_offsets.size() - 1;
    if (assembly_paths.size() != n_assemblies) {
        throw std::invalid_argument("assembly_paths length must match the number of assemblies");
    }
    if (is_targets.size() != n_assemblies) {
        throw std::invalid_argument("is_targets length must match the number of assemblies");
    }
    if (config.windowsize == 0) {
        throw std::invalid_argument("windowsize must be greater than zero");
    }
    if (!(config.tar_weight >= 0.0 && config.tar_weight <= 1.0)) {
        throw std::invalid_argument("tar_weight must be between 0 and 1");
    }
    if (!(config.neg_rank >= 0.0 && config.neg_rank <= 1.0)) {
        throw std::invalid_argument("neg_rank must be between 0 and 1");
    }
    if (!(config.edge_w_th_mul >= 0.0 && config.edge_w_th_mul <= 1.0)) {
        throw std::invalid_argument("edge_w_th_mul must be between 0 and 1");
    }
    if (!std::isfinite(config.consec_kmer_mul) || config.consec_kmer_mul <= 0.0) {
        throw std::invalid_argument("consec_kmer_mul must be finite and greater than zero");
    }
    if (config.max_nodes_cap && *config.max_nodes_cap == 0) {
        throw std::invalid_argument("max_nodes_cap must be greater than zero");
    }
    if (config.fasta_path.has_value() != config.csv_path.has_value()) {
        throw std::invalid_argument("Both fasta_path and csv_path are required for signature output");
    }

    FilteredGraph filtered;
    filtered.total_tar = std::count(is_targets.begin(), is_targets.end(), true);
    filtered.total_neg = is_targets.size() - filtered.total_tar;
    if (filtered.total_tar == 0) {
        throw std::invalid_argument("is_targets must contain at least one target assembly");
    }
    if (filtered.total_neg == 0) {
        throw std::invalid_argument("is_targets must contain at least one non-target assembly");
    }
    if (config.n_tar_th && *config.n_tar_th > filtered.total_tar) {
        throw std::invalid_argument("n_tar_th exceeds the number of target assemblies");
    }
    if (config.n_neg_th && *config.n_neg_th > filtered.total_neg) {
        throw std::invalid_argument("n_neg_th exceeds the number of non-target assemblies");
    }

    calculate_thresholds(
        filtered.total_tar,
        config,
        filtered
    );
    internal::ThreadPool pool(config.n_cpu);

    internal::log_python(" - Calculating target prevalence of graph nodes...");
    auto target_counts = internal::count_target_nodes(
        graph.nodes.prevalences,
        graph.assembly_nodes,
        graph.node_offsets,
        is_targets,
        filtered.total_tar,
        filtered.total_neg,
        pool
    );

    internal::log_python(" - Filtering graph and determining prevalence thresholds...");
    auto graph_topology = internal::prune_graph(
        graph.nodes.prevalences,
        graph.edges,
        target_counts,
        filtered.total_tar,
        filtered.total_neg,
        filtered.edge_weight_th,
        config.tar_weight,
        config.n_tar_th,
        filtered,
        pool
    );
    target_counts.reset();
    filtered.n_neg_th = neg_threshold(
        filtered.nodes,
        filtered.n_tar_th,
        config.neg_rank,
        config.n_neg_th
    );
    internal::log_python(
        " - Removed " + std::to_string(graph.edges.size() - filtered.edges.size()) +
        " edges failing weight/target prevalence thresholds, " +
        std::to_string(filtered.edges.size()) + " edges left"
    );
    internal::log_python(
        " - Removed " + std::to_string(graph.nodes.prevalences.size() - filtered.nodes.size()) +
        " isolated nodes, " + std::to_string(filtered.nodes.size()) + " nodes left"
    );
    internal::log_python(
        " - Target prevalence threshold: " + std::to_string(filtered.n_tar_th) +
        (config.n_tar_th.has_value() ? " (user-specified)" : " (estimated)")
    );
    internal::log_python(
        " - Non-target prevalence threshold: " + std::to_string(filtered.n_neg_th) +
        (config.n_neg_th.has_value() ? " (user-specified)" : " (estimated)")
    );

    internal::log_python(" - Growing subgraphs...");
    internal::grow_subgraphs(
        graph_topology,
        filtered.nodes,
        filtered.total_tar,
        filtered.n_tar_th,
        filtered.n_neg_th,
        filtered.min_nodes,
        filtered.max_nodes,
        filtered
    );
    graph_topology.reset();
    if (filtered.subgraphs.empty()) {
        throw std::runtime_error("No subgraphs found. Try relaxing the prevalence thresholds");
    }
    internal::log_python(" - Found " + std::to_string(filtered.subgraphs.size()) + " subgraphs");

    internal::log_python(" - Finding a representative for each subgraph...");
    auto signatures = internal::extract_signatures(
        filtered.subgraphs,
        graph.nodes.kmers,
        graph.nodes.offsets,
        graph.record_offsets,
        assembly_paths,
        is_targets,
        filtered.total_tar,
        config.kmerlen,
        config.windowsize,
        config.min_len,
        config.consec_kmer_mul,
        config.fasta_path,
        config.csv_path,
        pool
    );

    return {std::move(filtered), std::move(signatures)};
}

} // namespace seqwin
