/*
 * cuda_assert.h
 *
 *  Created on: 23-07-2016
 *      Author: francisco
 */

#ifndef CUDA_ASSERT_H_
#define CUDA_ASSERT_H_

#include <cuda_runtime.h>

#include <cstdio>

namespace particlesystem {

inline cudaError_t gpuAssert(cudaError_t result, const char *file, int line,
                             bool abort=true)
{
#ifndef NDEBUG
	if (result != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n",
				cudaGetErrorString(result), file, line);
		if (abort) exit(result);
	}
#endif
	return result;
}
#define cudaCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }

} // namespace particlesystem

#endif /* CUDA_ASSERT_H_ */
