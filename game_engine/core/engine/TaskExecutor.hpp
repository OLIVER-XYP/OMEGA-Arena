#ifndef TASKEXECUTOR_HPP
#define TASKEXECUTOR_HPP

#include <atomic>
#include <cstddef>
#include <functional>
#include <memory>
#include <string>

namespace hlt {

struct ExecutorRunStats {
    std::size_t parallel_for_calls{};
    std::size_t total_items_processed{};
};

class TaskExecutor {
public:
    virtual ~TaskExecutor() = default;
    virtual void parallel_for(std::size_t begin,
                              std::size_t end,
                              const std::function<void(std::size_t)> &fn) = 0;
    virtual std::string executor_name() const = 0;
    virtual std::size_t executor_thread_count() const = 0;
    virtual ExecutorRunStats run_stats() const = 0;
    virtual void reset_run_stats() = 0;
};

class InlineExecutor final : public TaskExecutor {
    std::atomic<std::size_t> parallel_for_calls_{0};
    std::atomic<std::size_t> total_items_processed_{0};

public:
    void parallel_for(std::size_t begin,
                      std::size_t end,
                      const std::function<void(std::size_t)> &fn) override {
        parallel_for_calls_.fetch_add(1, std::memory_order_relaxed);
        total_items_processed_.fetch_add(end >= begin ? (end - begin) : 0, std::memory_order_relaxed);
        for (std::size_t index = begin; index < end; ++index) {
            fn(index);
        }
    }

    std::string executor_name() const override { return "inline"; }

    std::size_t executor_thread_count() const override { return 1; }

    ExecutorRunStats run_stats() const override {
        return ExecutorRunStats{parallel_for_calls_.load(std::memory_order_relaxed),
                                total_items_processed_.load(std::memory_order_relaxed)};
    }

    void reset_run_stats() override {
        parallel_for_calls_.store(0, std::memory_order_relaxed);
        total_items_processed_.store(0, std::memory_order_relaxed);
    }
};

class ThreadPoolExecutor final : public TaskExecutor {
    class Impl;
    std::size_t thread_count_{};
    std::unique_ptr<Impl> impl;
    std::atomic<std::size_t> parallel_for_calls_{0};
    std::atomic<std::size_t> total_items_processed_{0};

public:
    explicit ThreadPoolExecutor(std::size_t thread_count = 0);
    ~ThreadPoolExecutor() override;

    ThreadPoolExecutor(const ThreadPoolExecutor &) = delete;
    ThreadPoolExecutor &operator=(const ThreadPoolExecutor &) = delete;

    void parallel_for(std::size_t begin,
                      std::size_t end,
                      const std::function<void(std::size_t)> &fn) override;

    std::string executor_name() const override { return "threadpool"; }

    std::size_t executor_thread_count() const override { return thread_count_; }

    ExecutorRunStats run_stats() const override {
        return ExecutorRunStats{parallel_for_calls_.load(std::memory_order_relaxed),
                                total_items_processed_.load(std::memory_order_relaxed)};
    }

    void reset_run_stats() override {
        parallel_for_calls_.store(0, std::memory_order_relaxed);
        total_items_processed_.store(0, std::memory_order_relaxed);
    }
};

} // namespace hlt

#endif // TASKEXECUTOR_HPP
