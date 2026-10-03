#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace seqwin {

/**
 * @brief Fixed-size owning array that avoids value-initializing elements.
 *
 * Unlike `std::vector<T>(n)` or `std::make_unique<T[]>(n)`, this class allocates
 * with `new T[n]`. For scalar and trivially default-initialized element types,
 * this avoids value-initializing every element, which can be expensive for very
 * large arrays.
 *
 * Important:
 * - Every element must be assigned before it is read.
 * - This is not a full `std::vector` replacement.
 * - It intentionally provides no `resize()`, `reserve()`, `push_back()`, or `capacity()`.
 */
template <typename T>
class NoInitArray {
public:
    NoInitArray() noexcept = default;

    explicit NoInitArray(std::size_t size)
        : size_(size)
        , data_(size == 0 ? nullptr : new T[size])
    {}

    NoInitArray(const NoInitArray&) = delete;
    NoInitArray& operator=(const NoInitArray&) = delete;

    NoInitArray(NoInitArray&& other) noexcept
        : size_(other.size_)
        , data_(std::move(other.data_))
    {
        other.size_ = 0;
    }

    NoInitArray& operator=(NoInitArray&& other) noexcept
    {
        if (this != &other) {
            data_ = std::move(other.data_);
            size_ = other.size_;
            other.size_ = 0;
        }
        return *this;
    }

    std::size_t size() const noexcept { return size_; }
    bool empty() const noexcept { return size_ == 0; }

    T* data() noexcept { return data_.get(); }
    const T* data() const noexcept { return data_.get(); }
    T* begin() noexcept { return data_.get(); }
    T* end() noexcept { return data_.get() + size_; }
    const T* begin() const noexcept { return data_.get(); }
    const T* end() const noexcept { return data_.get() + size_; }
    const T* cbegin() const noexcept { return data_.get(); }
    const T* cend() const noexcept { return data_.get() + size_; }

    T& operator[](std::size_t i) noexcept { return data_[i]; }
    const T& operator[](std::size_t i) const noexcept { return data_[i]; }

    void swap(NoInitArray& other) noexcept
    {
        std::swap(size_, other.size_);
        std::swap(data_, other.data_);
    }

    friend void swap(NoInitArray& a, NoInitArray& b) noexcept { a.swap(b); }

    void reset() noexcept
    {
        data_.reset();
        size_ = 0;
    }

private:
    std::size_t size_ = 0;
    std::unique_ptr<T[]> data_;
};

/**
 * @brief Location metadata for a minimizer.
 */
struct Kmer {
    /** 0-based position of the minimizer within its FASTA record. */
    std::uint32_t pos;
    /** 0-based global index of the FASTA record. */
    std::uint32_t record_idx;
};

/**
 * @brief Minimizer graph node for one unique minimizer hash.
 *
 * `start` is the beginning of this node's entries in `Graph.kmers`. The end
 * is the next node's `start`, or `Graph.kmers.size()` for the final node.
 */
struct Node {
    /** Hash value of the minimizers represented by this node. */
    std::uint64_t hash;
    /** Start of this node's minimizer entries in `Graph.kmers`. */
    std::size_t start;
    /** Number of assemblies containing this node's minimizer. */
    std::size_t prevalence;
};

/**
 * @brief Undirected weighted edge between two graph nodes.
 * Endpoints are indices into the graph's node array `Graph.nodes`.
 */
struct Edge {
    /** Index of the smaller endpoint in the graph's node array. */
    std::size_t first;
    /** Index of the larger endpoint in the graph's node array. */
    std::size_t second;
    /** Number of assemblies where the endpoints are adjacent. */
    std::size_t weight;
};

/**
 * @brief Container for the minimizer graph returned by `build()`.
 */
struct Graph {
    /**
     * Minimizer occurrences in all assemblies, grouped and sorted by hash.
     * `record_idx` is nondecreasing within each node range.
     */
    NoInitArray<Kmer> kmers;
    /** Sorted by hash. */
    NoInitArray<Node> nodes;
    /** Indexed into `nodes`; sorted by descending weight, then ascending endpoints. */
    NoInitArray<Edge> edges;
    /** Cumulative global FASTA record offsets by assembly. */
    std::vector<std::uint32_t> record_offsets;
    /** FASTA record IDs in global record order. */
    std::vector<std::string> record_ids;
    /**
     * @brief Node indices grouped by assembly.
     *
     * For assembly `i`, `[node_offsets[i], node_offsets[i + 1])` contains
     * unique nodes present in that assembly, in ascending node-index order.
     */
    NoInitArray<std::size_t> assembly_nodes;
    /** Cumulative offsets into `assembly_nodes` by assembly. */
    std::vector<std::size_t> node_offsets;
};

} // namespace seqwin
