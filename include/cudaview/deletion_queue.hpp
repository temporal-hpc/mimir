#pragma once

#include <functional>
#include <vector>

struct DeletionQueue
{
    std::vector<std::function<void()>> deletors;

    void pushFunction(std::function<void()>&& function)
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
