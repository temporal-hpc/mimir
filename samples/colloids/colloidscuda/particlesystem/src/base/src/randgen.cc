#include "base/randgen.h"

#include <curand.h>

#include <iostream>

#include "base/cuda_check.h"

namespace particlesystem {

template <typename T>
void generateUniform(curandGenerator_t gen, T *d_out, size_t num_elements);

template <>
void generateUniform<double>(curandGenerator_t gen, double* d_out, size_t n)
{
	curandGenerateUniformDouble(gen, d_out, n);
}

template <>
void generateUniform<float>(curandGenerator_t gen, float* d_out, size_t n)
{
	curandGenerateUniform(gen, d_out, n);
}

template <typename T>
RandGen<T>::RandGen(unsigned int max_idx, unsigned long long seed):
	index_(0),
	maxidx_(max_idx),
	data_(nullptr)
{
	curandGenerator_t dGenerator;
	data_ = new T[maxidx_];

	T *dData = nullptr;
	cudaCheck(cudaMalloc(&dData, max_idx * sizeof(T)));

	curandCreateGenerator(&dGenerator, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(dGenerator, seed);
	generateUniform<T>(dGenerator, dData, max_idx);
	curandDestroyGenerator(dGenerator);

	cudaCheck(cudaMemcpy(data_, dData, max_idx * sizeof(T),
			             cudaMemcpyDeviceToHost));
	cudaCheck(cudaFree(dData));

    std::cout << "[INFO] Initializing RNG with " << max_idx << " values."
    		  << std::endl;
}

template <typename T>
RandGen<T>::~RandGen()
{
	std::cout << "[INFO] Destroying RNG: " << index_ << " values were used."
			  << std::endl;
	delete [] data_;
}

template <typename T>
T RandGen<T>::next()
{
// TODO: Remove maxidx_ variable and do proper exception handling instead.
	if (index_ >= maxidx_)
	{
		std::cerr << "[ERROR] The random number generator has ran out."
				  << std::endl;
		return 0;
	}
	return data_[index_++];
}

template class RandGen<float>;
template class RandGen<double>;

} // namespace particlesystem
