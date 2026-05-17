#pragma once

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace seqwin {

class ThreadPool {
public:
    explicit ThreadPool(std::size_t n_workers)
        : n_workers_(std::max<std::size_t>(1, n_workers))
    {
        workers_.reserve(n_workers_);
        for (std::size_t i = 0; i < n_workers_; ++i) {
            workers_.emplace_back([this, i]() {
                worker_loop(i);
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

    template <typename Fn>
    void parallel_for(std::size_t n_items, Fn&& fn)
    {
        if (n_items == 0) {
            return;
        }

        const std::size_t active_workers = std::min(n_workers_, n_items);
        const std::size_t chunk_size = (n_items + active_workers - 1) / active_workers;

        auto shared_fn = std::make_shared<std::function<void(std::size_t, std::size_t, std::size_t)>>(std::forward<Fn>(fn));

        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_done_.wait(lock, [this]() { return pending_tasks_ == 0; });
            current_exception_ = nullptr;
            epoch_ += 1;
            current_epoch_ = epoch_;
            pending_tasks_ = 0;

            for (std::size_t worker_id = 0; worker_id < active_workers; ++worker_id) {
                const std::size_t start = worker_id * chunk_size;
                if (start >= n_items) {
                    break;
                }
                const std::size_t end = std::min(start + chunk_size, n_items);
                ++pending_tasks_;
                tasks_.push_back(Task{start, end, worker_id, shared_fn, current_epoch_});
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
        std::size_t worker_id;
        std::shared_ptr<std::function<void(std::size_t, std::size_t, std::size_t)>> fn;
        std::size_t epoch;
    };

    void worker_loop(std::size_t worker_index)
    {
        (void)worker_index;
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
                (*task.fn)(task.start, task.end, task.worker_id);
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

} // namespace seqwin
