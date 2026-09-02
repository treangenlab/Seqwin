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

#include <ankerl/unordered_dense.h>

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

} // namespace

FilterResult filter(
    const Kmer* kmers, Node* nodes, std::size_t n_nodes,
    const Edge* edges, std::size_t n_edges, const std::uint32_t* record_offsets,
    std::size_t n_record_offsets, const bool* is_targets, std::size_t n_assemblies,
    const double* jaccard, std::size_t jaccard_rows, std::size_t jaccard_cols,
    const FilterConfig& config
) {
    internal::log_python(" - Calculating node penalty scores...");
    get_penalty(kmers, nodes, n_nodes, record_offsets, n_record_offsets,
                is_targets, n_assemblies, config.n_cpu);
    std::size_t n_targets = 0;
    std::size_t n_non_targets = 0;
    for (std::size_t i = 0; i < n_assemblies; ++i) {
        is_targets[i] ? ++n_targets : ++n_non_targets;
    }

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

    internal::log_python(" - Filtering graph edges and nodes...");
    ankerl::unordered_dense::set<std::uint64_t> connected;
    connected.reserve(n_nodes);
    std::vector<Edge> retained_edges;
    retained_edges.reserve(n_edges);
    const auto edge_weight_th_int = static_cast<std::size_t>(edge_weight_th);
    for (std::size_t i = 0; i < n_edges; ++i) {
        if (edges[i].weight > edge_weight_th_int) {
            retained_edges.push_back(edges[i]);
            connected.insert(edges[i].first);
            connected.insert(edges[i].second);
        }
    }
    std::vector<Node> retained_nodes;
    retained_nodes.reserve(n_nodes);
    for (std::size_t i = 0; i < n_nodes; ++i) if (connected.count(nodes[i].hash)) retained_nodes.push_back(nodes[i]);
    internal::log_python(" - Removed " + std::to_string(n_edges - retained_edges.size()) +
        " edges with weight<" + format_value(edge_weight_th, 3) + ", " +
        std::to_string(retained_edges.size()) + " edges left");
    internal::log_python(" - Removed " + std::to_string(n_nodes - retained_nodes.size()) +
        " isolated nodes, " + std::to_string(retained_nodes.size()) + " nodes left");

    std::vector<std::uint64_t> seeds;
    seeds.reserve(retained_nodes.size());
    for (const auto& node : retained_nodes) if (node.penalty <= penalty_th) seeds.push_back(node.hash);
    // One engine deliberately drives both shuffles, making the complete operation reproducible.
    std::mt19937_64 rng(config.seed);
    std::shuffle(seeds.begin(), seeds.end(), rng);
    internal::log_python(" - Expanding subgraphs from " + std::to_string(seeds.size()) +
        " seed nodes (penalty<=" + format_value(penalty_th, 5) + ")...");
    auto extracted = get_subgraphs(retained_nodes.data(), retained_nodes.size(),
        retained_edges.data(), retained_edges.size(), seeds, penalty_th, min_nodes,
        max_nodes.value_or(std::numeric_limits<std::size_t>::max()));
    if (extracted.first.empty()) {
        throw std::runtime_error("No low-penalty subgraph was found. Try decrease --stringency, or increase --penalty-th");
    }
    std::shuffle(extracted.first.begin(), extracted.first.end(), rng);
    internal::log_python(" - Found " + std::to_string(extracted.first.size()) + " low-penalty subgraphs");

    Graph compact = filter_kmers(kmers, retained_nodes.data(), retained_nodes.size(), std::move(extracted.second));
    ankerl::unordered_dense::set<std::uint64_t> final_hashes;
    final_hashes.reserve(compact.nodes.size());
    for (const auto& node : compact.nodes) final_hashes.insert(node.hash);
    std::vector<Edge> final_edges;
    final_edges.reserve(retained_edges.size());
    for (const auto& edge : retained_edges) {
        if (final_hashes.count(edge.first) && final_hashes.count(edge.second)) {
            final_edges.push_back(edge);
        }
    }
    internal::log_python(" - " + std::to_string(compact.kmers.size()) + " k-mers left");
    return {std::move(compact.kmers), std::move(compact.nodes), std::move(final_edges),
            std::move(extracted.first), penalty_th, edge_weight_th, min_nodes, max_nodes};
}

} // namespace seqwin
