#include "TaskExecutor.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace hlt {

namespace {

std::size_t read_env_size_t(const char *name, std::size_t default_value, std::size_t min_value) {
    const char *raw = std::getenv(name);
    if (raw == nullptr || raw[0] == '\0') {
        return default_value;
    }
    char *end_ptr = nullptr;
    const auto parsed = std::strtoul(raw, &end_ptr, 10);
    if (end_ptr == raw) {
        return default_value;
    }
    const std::size_t value = static_cast<std::size_t>(parsed);
    return std::max(min_value, value);
}

std::size_t small_task_threshold() {
    static const std::size_t value = read_env_size_t("HALITE_EXEC_SMALL_TASK_THRESHOLD", 96, 1);
    return value;
}

std::size_t chunk_factor() {
    static const std::size_t value = read_env_size_t("HALITE_EXEC_CHUNK_FACTOR", 4, 1);
    return value;
}

} // namespace

class ThreadPoolExecutor::Impl {
    std::mutex mutex;
    std::condition_variable condition;
    std::deque<std::function<void()>> tasks;
    std::vector<std::thread> workers;
    bool stopping = false;

    void worker_loop() {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(mutex);
                condition.wait(lock, [&] { return stopping || !tasks.empty(); });
                if (stopping && tasks.empty()) {
                    return;
                }
                task = std::move(tasks.front());
                tasks.pop_front();
            }
            task();
        }
    }

public:
    explicit Impl(std::size_t thread_count) {
        if (thread_count == 0) {
            thread_count = std::max<std::size_t>(1, std::thread::hardware_concurrency());
        }
        workers.reserve(thread_count);
        for (std::size_t i = 0; i < thread_count; ++i) {
            workers.emplace_back([this] { worker_loop(); });
        }
    }

    ~Impl() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
        }
        condition.notify_all();
        for (auto &worker : workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    void parallel_for(std::size_t begin,
                      std::size_t end,
                      std::size_t chunk_size,
                      const std::function<void(std::size_t)> &fn) {
        if (begin >= end) {
            return;
        }

        const std::size_t total_items = end - begin;
        const std::size_t safe_chunk_size = chunk_size == 0 ? 1 : chunk_size;
        const std::size_t chunk_count = (total_items + safe_chunk_size - 1) / safe_chunk_size;

        std::mutex wait_mutex;
        std::condition_variable wait_condition;
        std::atomic<std::size_t> remaining_chunks{chunk_count};

        for (std::size_t chunk_index = 0; chunk_index < chunk_count; ++chunk_index) {
            const std::size_t chunk_begin = begin + chunk_index * safe_chunk_size;
            const std::size_t chunk_end = std::min(end, chunk_begin + safe_chunk_size);
            {
                std::lock_guard<std::mutex> lock(mutex);
                tasks.emplace_back([&, chunk_begin, chunk_end] {
                    for (std::size_t index = chunk_begin; index < chunk_end; ++index) {
                        fn(index);
                    }
                    if (remaining_chunks.fetch_sub(1) == 1) {
                        std::lock_guard<std::mutex> done_lock(wait_mutex);
                        wait_condition.notify_one();
                    }
                });
            }
            condition.notify_one();
        }

        std::unique_lock<std::mutex> lock(wait_mutex);
        wait_condition.wait(lock, [&] { return remaining_chunks.load() == 0; });
    }
};

ThreadPoolExecutor::ThreadPoolExecutor(std::size_t thread_count)
    : thread_count_(thread_count == 0 ? std::max<std::size_t>(1, std::thread::hardware_concurrency()) : thread_count),
      impl(std::make_unique<Impl>(thread_count_)) {}

ThreadPoolExecutor::~ThreadPoolExecutor() = default;

void ThreadPoolExecutor::parallel_for(std::size_t begin,
                                      std::size_t end,
                                      const std::function<void(std::size_t)> &fn) {
    parallel_for_calls_.fetch_add(1, std::memory_order_relaxed);
    total_items_processed_.fetch_add(end >= begin ? (end - begin) : 0, std::memory_order_relaxed);

    if (begin >= end) {
        return;
    }

    const std::size_t total_items = end - begin;
    if (thread_count_ <= 1 || total_items <= small_task_threshold()) {
        for (std::size_t index = begin; index < end; ++index) {
            fn(index);
        }
        return;
    }

    const std::size_t chunk_size = std::max<std::size_t>(1, total_items / (thread_count_ * chunk_factor()));
    impl->parallel_for(begin, end, chunk_size, fn);
}

} // namespace hlt
