#include "base/kernel_util.h"

#include "base/cuda_check.h"

namespace particlesystem {

////////////////////////////////////////////////////////////////////////////////
/// Kernel -- init Array
////////////////////////////////////////////////////////////////////////////////

template<typename T>
__global__ void
memsetArrayKernel(T *d_arr, const T value, const size_t array_size)
{
    int tidx = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(; tidx < array_size; tidx += stride)
    {
        d_arr[tidx] = value;
    }
}

template<typename T>
void memsetArray(T *d_arr, const T value, const size_t array_size,
                 uint2 launchData)
{
	memsetArrayKernel<T> <<< launchData.x, launchData.y >>>
	(d_arr, value, array_size);
	cudaCheck(cudaDeviceSynchronize());
}

////////////////////////////////////////////////////////////////////////////////
/// Kernel -- Init two arrays
////////////////////////////////////////////////////////////////////////////////

template<typename T>
__global__ void
memsetDualArraysKernel(T *d_arr1, T *d_arr2, const T value, const size_t n)
{
    int tidx = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(; tidx < n; tidx += stride)
    {
        d_arr1[tidx] = value;
        d_arr2[tidx] = value;
    }
}

template<typename T>
void memsetDualArrays(T *d_arr1, T *d_arr2, const T value, const size_t n,
                      uint2 launchData)
{
	memsetDualArraysKernel<T> <<< launchData.x, launchData.y >>>
	(d_arr1, d_arr2, value, n);
	cudaCheck(cudaDeviceSynchronize());
}

////////////////////////////////////////////////////////////////////////////////
/// Kernel -- Copy positions
////////////////////////////////////////////////////////////////////////////////

__global__ void
copyPositionsKernel(double2 *d_dst, const double2 *d_src, const size_t n)
{
    int tidx = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(; tidx < n; tidx += stride)
    {
        d_dst[tidx] = d_src[tidx];
    }
}

void copyPositions(double2 *d_dst, double2 *d_src, const size_t array_size,
                   uint2 launchData)
{
	copyPositionsKernel <<< launchData.x, launchData.y >>>
	(d_dst, d_src, array_size);
	cudaCheck(cudaDeviceSynchronize());
}

template void memsetArray<int>(int *d_arr, const int value,
                               const size_t array_size, uint2 launchData);
template void memsetArray<double>(double *d_arr, const double value,
                                  const size_t array_size, uint2 launchData);
template void memsetDualArrays<int>(int *d_arr1, int *d_arr2, const int value,
		                            const size_t n, uint2 launchData);

} // namespace particlesystem
