#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
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
 * @li Every element must be assigned before it is read.
 * @li This is not a full `std::vector` replacement.
 * @li It intentionally provides no `resize()`, `reserve()`, `push_back()`, or `capacity()`.
 */
template <typename T>
class NoInitArray {
public:
    using value_type = T;

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
 * @brief Parallel arrays representing nodes in the final minimizer graph.
 *
 * Each node represents a unique minimizer hash. Nodes are ordered by ascending hash.
 *
 * The arrays share the same node indices. For node `i`,
 * `kmers[offsets[i]:offsets[i + 1]]` contains its minimizer occurrences,
 * `prevalences[i]` is its assembly prevalence, and `hashes[i]` is its hash.
 */
struct Nodes {
    /** Minimizer occurrences grouped by node, ordered by record index and position. */
    NoInitArray<Kmer> kmers;
    /** Cumulative offsets into `kmers` by node, including the final sentinel. */
    NoInitArray<std::size_t> offsets;
    /** Number of assemblies containing each node's minimizer. */
    NoInitArray<std::uint32_t> prevalences;
    /** Unique minimizer hashes in strictly ascending order. */
    NoInitArray<std::uint64_t> hashes;

    std::size_t size() const noexcept {
        return hashes.size();
    }
};

/**
 * @brief Weighted, undirected edge between two graph nodes.
 * Endpoints are node indices into the graph's node arrays.
 */
struct Edge {
    /** Index of the smaller endpoint. */
    std::size_t first;
    /** Index of the larger endpoint. */
    std::size_t second;
    /** Number of assemblies where the endpoints are adjacent. */
    std::size_t weight;
};

/**
 * @brief Container for the minimizer graph returned by `build()`.
 *
 * `Graph` members do not encode semantic defaults. Build and merge routines
 * construct owned containers separately and transfer them into the final graph
 * with move assignment or move construction.
 */
struct Graph {
    /** Node arrays sharing the same node indices. */
    Nodes nodes;
    /** Endpoints are node indices; sorted by descending weight, then ascending endpoints. */
    NoInitArray<Edge> edges;
    /** Cumulative global FASTA record offsets by assembly, including the final sentinel. */
    std::vector<std::uint32_t> record_offsets;
    /**
     * Unique node indices grouped by assembly.
     * For assembly `i`, `assembly_nodes[node_offsets[i]:node_offsets[i + 1]]`
     * contains unique nodes present in that assembly, in ascending node-index order.
     */
    NoInitArray<std::size_t> assembly_nodes;
    /** Cumulative offsets into `assembly_nodes` by assembly, including the final sentinel. */
    std::vector<std::size_t> node_offsets;
};

} // namespace seqwin
