/*
 * kernel_random.cuh
 *
 *  Created on: 22-07-2016
 *      Author: francisco
 */

#ifndef KERNEL_RANDOM_CUH_
#define KERNEL_RANDOM_CUH_

#include <curand_kernel.h>

namespace particlesystem {

__device__ __host__ double
clamp(double value, double lower, double upper);

__device__ double2
gaussianNoise(curandState* global_states, unsigned int idx);

__global__ void
setupRngKernel(curandState* random_states, unsigned long long seed,
			   unsigned int num_elements);

// TODO: Change launch data to grid, block or a data structure
void setupRng(curandState* d_random_states, unsigned long long seed,
              unsigned int num_elements, uint2 pLaunchData);

} // namespace particlesystem

#endif /* KERNEL_RANDOM_CUH_ */
