#include "thread_pool.h"

namespace lattice {

ThreadPool::ThreadPool(uint32_t num_threads) {
    if (num_threads == 0) {
        num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0) num_threads = 4; // fallback
    }

    workers_.reserve(num_threads);
    for (uint32_t i = 0; i < num_threads; ++i) {
        workers_.emplace_back([this] {
            // Worker loop: run forever until stop_ is set and queue is empty.
            while (true) {
                std::function<void()> task;

                {
                    // Acquire the lock and wait for either:
                    //   (a) a task to appear in the queue, or
                    //   (b) the stop flag to be set
                    std::unique_lock<std::mutex> lock(mutex_);
                    cv_.wait(lock, [this] {
                        return stop_ || !tasks_.empty();
                    });

                    // If stopping and no tasks left, exit the thread.
                    if (stop_ && tasks_.empty()) return;

                    // Pop the next task.
                    task = std::move(tasks_.front());
                    tasks_.pop();
                }

                // Execute outside the lock so other workers can grab tasks.
                task();
            }
        });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }
    // Wake ALL workers so they see the stop flag and exit.
    cv_.notify_all();

    for (auto& worker : workers_) {
        worker.join();  // Wait for each thread to finish.
    }
}

} // namespace lattice
