#pragma once

#include <functional> // std::function
#include <vector> // std::vector

struct DeletionQueue
{
    std::vector<std::function<void()>> deletors;

    void add(std::function<void()>&& function)
    {
        deletors.push_back(function);
    }

    void flush()
    {
        for (auto it = deletors.rbegin(); it != deletors.rend(); ++it)
        {
            (*it)();
        }
        deletors.clear();
    }
};

static_assert(std::is_default_constructible_v<DeletionQueue>);
static_assert(std::is_nothrow_default_constructible_v<DeletionQueue>);