#include "seqwin/filter.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace seqwin {

Graph filter_kmers(
    const Kmer* kmers,
    const Node* nodes,
    std::size_t n_nodes,
    std::vector<std::uint64_t> used_hashes
) {
    std::sort(used_hashes.begin(), used_hashes.end());

    Graph graph;

    std::vector<std::size_t> used_node_indices;
    used_node_indices.reserve(used_hashes.size());

    std::size_t n_kmers = 0;
    std::size_t node_i = 0;
    std::size_t used_i = 0;
    while (node_i < n_nodes && used_i < used_hashes.size()) {
        const auto node_hash = nodes[node_i].hash;
        const auto used_hash = used_hashes[used_i];

        if (node_hash < used_hash) {
            ++node_i;
            continue;
        }
        if (used_hash < node_hash) {
            ++used_i;
            continue;
        }

        used_node_indices.push_back(node_i);
        n_kmers += nodes[node_i].stop - nodes[node_i].start;
        ++node_i;
        ++used_i;
    }

    graph.nodes = NoInitArray<Node>(used_node_indices.size());
    graph.kmers = NoInitArray<Kmer>(n_kmers);

    std::size_t new_start = 0;
    for (std::size_t out_node_i = 0; out_node_i < used_node_indices.size(); ++out_node_i) {
        const auto in_node_i = used_node_indices[out_node_i];
        const Node& old_node = nodes[in_node_i];

        const auto old_start = old_node.start;
        const auto old_stop = old_node.stop;
        const auto size = old_stop - old_start;

        Node new_node = old_node;
        new_node.start = new_start;
        new_node.stop = new_start + size;
        graph.nodes[out_node_i] = new_node;

        for (std::size_t k = 0; k < size; ++k) {
            const auto out_i = new_start + k;
            const auto in_i = old_start + k;
            graph.kmers[out_i] = kmers[in_i];
        }

        new_start += size;
    }

    return graph;
}

} // namespace seqwin
