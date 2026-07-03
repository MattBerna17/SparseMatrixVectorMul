#pragma once

#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

class ThreadPool {
public:
    std::uint64_t num_threads;

    // creates a thread pool with the given the number of threads
    ThreadPool(std::uint64_t num_threads = std::thread::hardware_concurrency()) {
        this->num_threads = num_threads;
        active_tasks_ = 0;
        
        for (size_t i = 0; i < num_threads; ++i) {
            threads_.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        // locking the queue so that data can be accessed safely
                        std::unique_lock<std::mutex> lock(queue_mutex_);

                        // waiting until there is a task to execute or the pool is stopped
                        cv_.wait(lock, [this] {
                            return !tasks_.empty() || stop_;
                        });

                        // exit the thread in case the pool is stopped and there are no tasks
                        if (stop_ && tasks_.empty()) {
                            return;
                        }

                        // get the next task from the queue
                        task = std::move(tasks_.front());
                        tasks_.pop();
                    }

                    // execute the task
                    task();
                    
                    // signal end of task
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex_);
                        active_tasks_--;
                        // if no other job waiting, signal on barrier
                        if (active_tasks_ == 0) {
                            barrier_cv_.notify_all();
                        }
                    }
                }
            });
        }
    }

    // destructor which terminates the pool
    ~ThreadPool() {
        {
            // lock the queue to update the stop flag safely
            std::unique_lock<std::mutex> lock(queue_mutex_);
            stop_ = true;
        }

        // notify all threads
        cv_.notify_all();

        // join all worker threads to ensure they have completed their tasks
        for (auto& thread : threads_) {
            thread.join();
        }
    }

    // wait until all the enqueued tasks are completed
    void barrier() {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        // wait until no more active tasks
        barrier_cv_.wait(lock, [this] {
            return active_tasks_ == 0;
        });
    }

    // enqueue task for execution by the thread pool
    void enqueue(std::function<void()> task) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            tasks_.emplace(std::move(task));
            active_tasks_++; // increment task counter for each new job
        }
        cv_.notify_one();
    }

private:
    // vector to store worker threads
    std::vector<std::thread> threads_;

    // queue of tasks
    std::queue<std::function<void()>> tasks_;

    // mutex to synchronize access to shared data
    std::mutex queue_mutex_;

    // // mutex to synchronize access to the active_tasks_ variable
    // std::mutex count_mutex_;

    // cv to signal changes in the state of the tasks queue
    std::condition_variable cv_;
    
    // cv to signal when all tasks are done (for the barrier)
    std::condition_variable barrier_cv_;

    // counter for active and queued tasks
    size_t active_tasks_;

    // flag to indicate whether the thread pool should stop or not
    bool stop_ = false;
};