#include "seqwin/filter.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
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

double expected_frac(
    const double* jaccard,
    std::size_t n,
    const bool* is_targets,
    bool rows_are_targets
) {
    double sum = 0.0;
    std::size_t count = 0;
    for (std::size_t row = 0; row < n; ++row) {
        if (is_targets[row] != rows_are_targets) continue;
        for (std::size_t col = 0; col < n; ++col) {
            if (!is_targets[col]) continue;
            const double value = jaccard[row * n + col];
            if (!std::isfinite(value) || value < 0.0 || value > 1.0) {
                throw std::invalid_argument("Jaccard values must be finite and between 0 and 1");
            }
            sum += 2.0 * value / (1.0 + value);
            ++count;
        }
    }
    if (count == 0) throw std::invalid_argument("Jaccard groups must not be empty");
    return sum / static_cast<double>(count);
}

FilterResult calculate_thresholds(
    const Node* nodes,
    std::size_t n_nodes,
    const bool* is_targets,
    std::size_t n_assemblies,
    const double* jaccard,
    std::size_t jaccard_rows,
    std::size_t jaccard_cols,
    const FilterConfig& config
) {
    const auto n_targets = static_cast<std::size_t>(
        std::count(is_targets, is_targets + n_assemblies, true)
    );
    const auto n_non_targets = n_assemblies - n_targets;

    double penalty_th;
    if (config.penalty_th) {
        penalty_th = *config.penalty_th;
        internal::log_python("Penalty threshold is provided (--penalty-th), skip auto estimation", "warning");
    } else {
        internal::log_python(" - Calculating penalty threshold...");
        double absence;
        double presence;
        if (jaccard) {
            if (jaccard_rows != n_assemblies || jaccard_cols != n_assemblies) {
                throw std::invalid_argument("Jaccard matrix shape must match the number of assemblies");
            }
            absence = 1.0 - expected_frac(jaccard, n_assemblies, is_targets, true);
            presence = expected_frac(jaccard, n_assemblies, is_targets, false);
        } else {
            double target_weight = 0.0;
            double absence_sum = 0.0;
            double presence_sum = 0.0;
            for (std::size_t i = 0; i < n_nodes; ++i) {
                const double weight = nodes[i].n_tar;
                target_weight += weight;
                absence_sum += (static_cast<double>(nodes[i].n_tar) / n_targets) * weight;
                presence_sum += (static_cast<double>(nodes[i].n_neg) / n_non_targets) * weight;
            }
            if (target_weight == 0.0) throw std::invalid_argument("No target minimizers are available for threshold estimation");
            absence = 1.0 - absence_sum / target_weight;
            presence = presence_sum / target_weight;
        }
        internal::log_python(" - Expected k-mer absence in targets: " +
            format_value(absence, 5));
        internal::log_python(" - Expected k-mer presence in non-targets: " +
            format_value(presence, 5));
        penalty_th = (1.0 - config.stringency / 10.0) * std::sqrt(absence * presence);
        internal::log_python(" - Calculated penalty threshold: " +
            format_value(penalty_th, 5));
        if (penalty_th > config.penalty_th_cap) {
            penalty_th = config.penalty_th_cap;
            internal::log_python(" - Calculated penalty threshold is too large (capped at " +
                format_value(penalty_th, 5) + ")", "warning");
        }
    }

    const double edge_weight_th = config.edge_w_th_mul * (1.0 - penalty_th) * n_targets;
    const std::size_t gap_len = (config.windowsize + 1) / 2;
    const std::size_t min_nodes = std::max(config.min_nodes_floor, config.min_len / gap_len + 1);
    const std::optional<std::size_t> max_nodes = config.max_len
        ? std::optional<std::size_t>(*config.max_len / gap_len + 1)
        : config.max_nodes_cap;
    if (max_nodes) {
        internal::log_python(" - Subgraph size limit is set to [" +
            std::to_string(min_nodes) + ", " + std::to_string(*max_nodes) + "]");
    } else {
        internal::log_python(" - Upper limit of subgraph size is not set. Lower limit is set to " +
            std::to_string(min_nodes), "warning");
    }

    FilterResult result;
    result.penalty_th = penalty_th;
    result.edge_weight_th = edge_weight_th;
    result.min_nodes = min_nodes;
    result.max_nodes = max_nodes;
    return result;
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
    internal::get_penalty(
        kmers, nodes, n_nodes, record_offsets, n_record_offsets, is_targets, n_assemblies, config.n_cpu
    );
    auto result = calculate_thresholds(
        nodes, n_nodes, is_targets, n_assemblies, jaccard, jaccard_rows, jaccard_cols, config
    );

    internal::log_python(" - Filtering graph edges and nodes...");
    auto pruned = internal::prune_graph(
        nodes, n_nodes, edges, n_edges, result.edge_weight_th
    );
    internal::log_python(" - Removed " + std::to_string(n_edges - pruned.edges.size()) +
        " edges with weight<" + format_value(result.edge_weight_th, 3) + ", " +
        std::to_string(pruned.edges.size()) + " edges left");
    internal::log_python(" - Removed " + std::to_string(n_nodes - pruned.nodes.size()) +
        " isolated nodes, " + std::to_string(pruned.nodes.size()) + " nodes left");

    std::mt19937_64 rng(config.seed);
    auto [subgraphs, used_nodes] = internal::get_subgraphs(
        pruned.nodes, pruned.edges, result.penalty_th, result.min_nodes,
        result.max_nodes.value_or(std::numeric_limits<std::size_t>::max()), rng
    );
    if (subgraphs.empty()) {
        throw std::runtime_error("No low-penalty subgraph was found. Try decrease --stringency, or increase --penalty-th");
    }
    std::shuffle(subgraphs.begin(), subgraphs.end(), rng);
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
