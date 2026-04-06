#pragma once

#include <cstdint>
#include <functional>
#include <future>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <thread>
#include <vector>

namespace lattice {

// A fixed-size thread pool that pre-creates N worker threads and dispatches
// tasks to them via a shared queue.
//
// How it works:
//   1. Constructor spawns N threads. Each one loops forever, waiting for work.
//   2. submit() wraps your function in a packaged_task, pushes it to the queue,
//      and wakes one sleeping worker via the condition variable.
//   3. A worker pops the task, executes it, and goes back to waiting.
//   4. submit() returns a std::future so the caller can get the result later.
//   5. Destructor sets a stop flag, wakes all workers, and joins them.
//
// This is the same pattern used in production systems (database connection
// pools, web server request handlers, game engine job systems).

class ThreadPool {
public:
    // num_threads = 0 means use hardware_concurrency (number of CPU cores).
    explicit ThreadPool(uint32_t num_threads = 0);
    ~ThreadPool();

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    // Submit a callable and return a future for its result.
    //
    // Usage:
    //   auto future = pool.submit([]{ return 42; });
    //   int result = future.get();  // blocks until task completes
    //
    // The template deduces the return type automatically.
    template <typename F, typename... Args>
    auto submit(F&& f, Args&&... args) -> std::future<std::invoke_result_t<F, Args...>> {
        using ReturnType = std::invoke_result_t<F, Args...>;

        // packaged_task wraps a callable and gives us a future for its result.
        auto task = std::make_shared<std::packaged_task<ReturnType()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );

        std::future<ReturnType> future = task->get_future();

        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stop_) {
                throw std::runtime_error("submit() called on stopped ThreadPool");
            }
            // Wrap in a void() function so the queue has a uniform type.
            tasks_.emplace([task]() { (*task)(); });
        }

        // Wake one sleeping worker to pick up the task.
        cv_.notify_one();
        return future;
    }

    uint32_t num_threads() const {
        return static_cast<uint32_t>(workers_.size());
    }

private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;

    std::mutex mutex_;
    std::condition_variable cv_;
    bool stop_ = false;
};

} // namespace lattice
