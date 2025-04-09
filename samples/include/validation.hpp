#pragma once

#include <cuda_runtime_api.h>

#include <cstdio> // stderr
#include <source_location> // std::source_location
#include <stdexcept> // std::throw

using srcloc = std::source_location;

constexpr void checkCuda(cudaError_t code, bool panic = true,
    srcloc src = srcloc::current())
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "CUDA assertion: %s in function %s at %s(%d)\n",
            cudaGetErrorString(code), src.function_name(), src.file_name(), src.line()
        );
        if (panic)
        {
            throw std::runtime_error("CUDA failure!");
        }
    }
}
