#include "base/kernel_random.cuh"

#include "base/cuda_check.h"

namespace particlesystem {

__device__ __host__
double clamp(double value, double lower, double upper)
{
	return value < lower ? lower : (value > upper ? upper : value);
}

__device__
double2 gaussianNoise(curandState* global_states, unsigned int idx)
{
	// Bring the corresponding state from global memory
	curandState local_state = global_states[idx];
	double2 r = curand_normal2_double(&local_state);
	// Send the updated state back to global memory
	global_states[idx] = local_state;
	r.x = clamp(r.x, -3.0, 3.0);
	r.y = clamp(r.y, -3.0, 3.0);
	return r;
}

__global__ void
setupRngKernel(curandState* rng_states, unsigned long long seed,
			   unsigned int num_elements)
{
	unsigned int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int stride = blockDim.x * gridDim.x;

	for(; tidx < num_elements; tidx += stride)
	{
		curand_init(seed, tidx, 0, &rng_states[tidx]);
	}
}

// TODO: Change launch data to grid, block or a data structure
void setupRng(curandState* d_rng_states, unsigned long long seed,
              unsigned int num_elements, uint2 pLaunchData)
{
	setupRngKernel<<< pLaunchData.x, pLaunchData.y >>>
	(d_rng_states, seed, num_elements);
	cudaCheck(cudaDeviceSynchronize());
}

} // namespace particlesystem
