#pragma once

#include <cstddef>
#include <cstdint>

#include "seqwin/graph.hpp"

namespace seqwin {

/**
 * @brief Read-only, non-owning view over a contiguous array.
 *
 * The underlying array must remain valid for the lifetime of the view.
 */
template <typename T>
class ArrayView {
public:
    ArrayView(const T* data, std::size_t size) noexcept
        : data_(data)
        , size_(size)
    {}

    const T* data() const noexcept { return data_; }
    std::size_t size() const noexcept { return size_; }
    bool empty() const noexcept { return size_ == 0; }

    const T& operator[](std::size_t i) const noexcept { return data_[i]; }
    const T* begin() const noexcept { return data_; }
    const T* end() const noexcept { return size_ == 0 ? data_ : data_ + size_; }

private:
    const T* data_;
    std::size_t size_;
};

/**
 * @brief Read-only, non-owning view of the parallel node arrays in `Nodes`.
 */
struct NodesView {
    ArrayView<Kmer> kmers;
    ArrayView<std::size_t> offsets;
    ArrayView<std::uint32_t> prevalences;
    ArrayView<std::uint64_t> hashes;

    std::size_t size() const noexcept {
        return hashes.size();
    }
};

/**
 * @brief Read-only, non-owning view of a k-mer graph.
 */
struct GraphView {
    NodesView nodes;
    ArrayView<Edge> edges;
    ArrayView<std::uint32_t> record_offsets;
    ArrayView<std::size_t> assembly_nodes;
    ArrayView<std::size_t> node_offsets;
};

/**
 * @brief View an owning graph without copying its arrays or transferring ownership.
 *
 * The graph's arrays must remain valid for the lifetime of the view.
 */
inline GraphView make_view(const Graph& graph) noexcept
{
    return {
        {
            {graph.nodes.kmers.data(), graph.nodes.kmers.size()},
            {graph.nodes.offsets.data(), graph.nodes.offsets.size()},
            {graph.nodes.prevalences.data(), graph.nodes.prevalences.size()},
            {graph.nodes.hashes.data(), graph.nodes.hashes.size()}
        },
        {graph.edges.data(), graph.edges.size()},
        {graph.record_offsets.data(), graph.record_offsets.size()},
        {graph.assembly_nodes.data(), graph.assembly_nodes.size()},
        {graph.node_offsets.data(), graph.node_offsets.size()}
    };
}

} // namespace seqwin
