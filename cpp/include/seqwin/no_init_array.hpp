#pragma once

#include <cstddef>
#include <memory>
#include <utility>

namespace seqwin {

/**
 * A fixed-size owning array.
 *
 * Unlike `std::vector<T>(n) or std::make_unique<T[]>(n)`, this class allocates
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

} // namespace seqwin
