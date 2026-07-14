#include "renorm/work_queue.h"

#include <atomic>
#include <cassert>
#include <iostream>
#include <thread>
#include <vector>

using namespace renorm;

int main()
{
    constexpr int PRODUCERS = 4;
    constexpr int CONSUMERS = 4;
    constexpr int TASKS_PER_PRODUCER = 50000;

    constexpr int TOTAL =
        PRODUCERS * TASKS_PER_PRODUCER;

    WorkQueue queue(1 << 16);

    std::atomic<int> produced{0};
    std::atomic<int> consumed{0};

    //----------------------------------------------------------
    // Producers
    //----------------------------------------------------------

    std::vector<std::thread> producers;

    for (int p = 0; p < PRODUCERS; ++p)
    {
        producers.emplace_back([&]()
        {
            for (int i = 0; i < TASKS_PER_PRODUCER;)
            {
                if (queue.push([&]()
                {
                    consumed.fetch_add(
                        1,
                        std::memory_order_relaxed);
                }))
                {
                    ++i;

                    produced.fetch_add(
                        1,
                        std::memory_order_relaxed);
                }
                else
                {
                    std::this_thread::yield();
                }
            }
        });
    }

    //----------------------------------------------------------
    // Consumers
    //----------------------------------------------------------

    std::vector<std::thread> consumers;

    for (int c = 0; c < CONSUMERS; ++c)
    {
        consumers.emplace_back([&]()
        {
            WorkQueue::Task task;

            while (consumed.load(std::memory_order_relaxed)
                   < TOTAL)
            {
                if (queue.pop(task))
                {
                    task();
                }
                else
                {
                    std::this_thread::yield();
                }
            }
        });
    }

    //----------------------------------------------------------

    for (auto& t : producers)
        t.join();

    for (auto& t : consumers)
        t.join();

    //----------------------------------------------------------

    assert(produced == TOTAL);
    assert(consumed == TOTAL);

    std::cout << "====================================\n";
    std::cout << "WORK QUEUE TEST PASSED\n";
    std::cout << "Produced : " << produced << '\n';
    std::cout << "Consumed : " << consumed << '\n';
    std::cout << "====================================\n";

    return 0;
}