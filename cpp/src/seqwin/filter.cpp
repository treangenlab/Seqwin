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

void calculate_thresholds(
    const FilterConfig& config,
    FilteredGraph& filtered
) {
    // Calculate edge weight threshold
    const double edge_weight_th = config.edge_w_th_mul * filtered.total_tar;

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

    filtered.edge_weight_th = edge_weight_th;
    filtered.min_nodes = min_nodes;
    filtered.max_nodes = max_nodes;
}

std::size_t neg_threshold(
    const FilteredGraph& filtered,
    const FilterConfig& config
) {
    if (config.n_neg_th) {
        return *config.n_neg_th;
    }
    std::vector<std::uint32_t> counts;
    for (const auto& node : filtered.nodes) {
        if (node.n_tar >= filtered.n_tar_th) {
            counts.push_back(node.n_neg);
        }
    }
    if (counts.empty()) {
        return 0;
    }
    const std::size_t index = std::max<std::size_t>(
        1,
        static_cast<std::size_t>(std::ceil(config.neg_quantile * counts.size()))
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
    if (config.fasta_path.has_value() != config.csv_path.has_value()) {
        throw std::invalid_argument("Both fasta_path and csv_path are required for signature output");
    }
    if (!std::isfinite(config.consec_kmer_mul) || config.consec_kmer_mul <= 0.0) {
        throw std::invalid_argument("consec_kmer_mul must be finite and greater than zero");
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

    if (config.n_tar_th && *config.n_tar_th > filtered.total_tar) {
        throw std::invalid_argument("n_tar_th exceeds the number of target assemblies");
    }
    if (config.n_neg_th && *config.n_neg_th > filtered.total_neg) {
        throw std::invalid_argument("n_neg_th exceeds the number of non-target assemblies");
    }
    if (!(config.stringency >= 0.0 && config.stringency <= 10.0)) {
        throw std::invalid_argument("stringency must be between 0 and 10");
    }
    if (!(config.neg_quantile >= 0.0 && config.neg_quantile <= 1.0)) {
        throw std::invalid_argument("neg_quantile must be between 0 and 1");
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
    calculate_thresholds(config, filtered);

    internal::log_python(" - Pruning graph and counting node prevalences...");
    auto graph_topology = internal::prune_graph(
        graph.nodes.prevalences,
        graph.edges,
        target_counts,
        config,
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

    filtered.n_neg_th = neg_threshold(filtered, config);
    internal::log_python(" - Target prevalence threshold: " + std::to_string(filtered.n_tar_th));
    internal::log_python(" - Non-target prevalence threshold: " + std::to_string(filtered.n_neg_th));
    internal::get_subgraphs(graph_topology, filtered);
    if (filtered.subgraphs.empty()) {
        throw std::runtime_error(
            "No subgraph met the prevalence thresholds and minimum size"
        );
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
