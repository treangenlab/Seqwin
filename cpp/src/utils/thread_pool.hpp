#pragma once

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace seqwin::internal {

/**
 * @brief Simple fixed-size worker pool for parallel item and range processing.
 *
 * Calls that submit work to the same pool must not overlap, and callbacks must
 * not recursively submit work to the same pool.
 */
class ThreadPool {
public:
    explicit ThreadPool(std::size_t n_workers)
        : n_workers_(std::max<std::size_t>(1, n_workers))
    {
        workers_.reserve(n_workers_);
        for (std::size_t i = 0; i < n_workers_; ++i) {
            workers_.emplace_back([this]() {
                worker_loop();
            });
        }
    }

    ~ThreadPool()
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
            cv_job_.notify_all();
        }
        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    std::size_t size() const noexcept { return n_workers_; }

    /**
     * @brief Run a function for every item in `[0, n_items)`.
     *
     * Work is scheduled in contiguous chunks, but the callable is invoked for each
     * item and receives either `(item_index)` or `(item_index, chunk_id)`. The
     * `chunk_id` identifies the logical partition assigned by the scheduler, not
     * necessarily the physical thread executing it. Multiple item callbacks can
     * therefore share the same `chunk_id`. Chunk IDs are contiguous from zero and
     * always less than `size()`.
     *
     * Any exception thrown while processing an item is captured and rethrown on
     * the calling thread.
     *
     * @tparam Fn Callable type.
     * @param n_items Number of items in the range.
     * @param fn Function invoked once for each item.
     */
    template <typename Fn>
    void parallel_for(std::size_t n_items, Fn&& fn)
    {
        using FnType = std::decay_t<Fn>;
        static_assert(
            std::is_invocable_v<FnType&, std::size_t> ||
            std::is_invocable_v<FnType&, std::size_t, std::size_t>,
            "parallel_for callback must accept (item_index) or (item_index, chunk_id)"
        );
        if (n_items == 0) {
            return;
        }

        auto item_fn = std::make_shared<FnType>(std::forward<Fn>(fn));
        parallel_for_chunks(n_items, [item_fn](
            std::size_t start, std::size_t end, std::size_t chunk_id
        ) {
            for (std::size_t i = start; i < end; ++i) {
                if constexpr (std::is_invocable_v<FnType&, std::size_t, std::size_t>) {
                    std::invoke(*item_fn, i, chunk_id);
                } else {
                    std::invoke(*item_fn, i);
                }
            }
        });
    }

    /**
     * @brief Run a function over contiguous chunks of `[0, n_items)`.
     *
     * The callable receives `(start, end, chunk_id)`, where `[start, end)` is a
     * non-empty contiguous partition. The `chunk_id` identifies that logical
     * scheduler partition and is not guaranteed to identify the physical thread
     * executing it. Chunk IDs are contiguous from zero and always less than
     * `size()`.
     *
     * Any exception thrown while processing a chunk is captured and rethrown on
     * the calling thread.
     *
     * @tparam Fn Callable type.
     * @param n_items Number of items in the range.
     * @param fn Function invoked once for each non-empty chunk.
     */
    template <typename Fn>
    void parallel_for_chunks(std::size_t n_items, Fn&& fn)
    {
        using FnType = std::decay_t<Fn>;
        static_assert(
            std::is_invocable_v<FnType&, std::size_t, std::size_t, std::size_t>,
            "parallel_for_chunks callback must accept (start, end, chunk_id)"
        );
        if (n_items == 0) {
            return;
        }

        const std::size_t n_chunks = std::min(n_workers_, n_items);
        const std::size_t base = n_items / n_chunks;
        const std::size_t rem = n_items % n_chunks;

        auto shared_fn = std::make_shared<std::function<void(std::size_t, std::size_t, std::size_t)>>(std::forward<Fn>(fn));

        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_done_.wait(lock, [this]() { return pending_tasks_ == 0; });
            current_exception_ = nullptr;
            epoch_ += 1;
            current_epoch_ = epoch_;
            pending_tasks_ = 0;

            for (std::size_t chunk_id = 0; chunk_id < n_chunks; ++chunk_id) {
                const std::size_t start = chunk_id * base + std::min(chunk_id, rem);
                const std::size_t end = start + base + (chunk_id < rem ? 1 : 0);
                ++pending_tasks_;
                tasks_.push_back(Task{start, end, chunk_id, shared_fn, current_epoch_});
            }
            cv_job_.notify_all();
        }

        std::unique_lock<std::mutex> lock(mutex_);
        cv_done_.wait(lock, [this]() { return pending_tasks_ == 0; });
        if (current_exception_) {
            std::rethrow_exception(current_exception_);
        }
    }

private:
    struct Task {
        std::size_t start;
        std::size_t end;
        std::size_t chunk_id;
        std::shared_ptr<std::function<void(std::size_t, std::size_t, std::size_t)>> fn;
        std::size_t epoch;
    };

    void worker_loop()
    {
        while (true) {
            Task task;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_job_.wait(lock, [this]() { return stopping_ || !tasks_.empty(); });
                if (stopping_ && tasks_.empty()) {
                    return;
                }
                task = std::move(tasks_.back());
                tasks_.pop_back();
            }

            try {
                (*task.fn)(task.start, task.end, task.chunk_id);
            } catch (...) {
                std::lock_guard<std::mutex> lock(mutex_);
                if (!current_exception_) {
                    current_exception_ = std::current_exception();
                }
            }

            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (task.epoch == current_epoch_ && pending_tasks_ > 0) {
                    --pending_tasks_;
                    if (pending_tasks_ == 0) {
                        cv_done_.notify_one();
                    }
                }
            }
        }
    }

    std::size_t n_workers_;
    std::vector<std::thread> workers_;
    std::vector<Task> tasks_;
    std::mutex mutex_;
    std::condition_variable cv_job_;
    std::condition_variable cv_done_;
    std::size_t pending_tasks_ = 0;
    std::size_t epoch_ = 0;
    std::size_t current_epoch_ = 0;
    std::exception_ptr current_exception_;
    bool stopping_ = false;
};

} // namespace seqwin::internal
