#include "base/simparameters.h"

#include <cmath>
#include <iostream>
#include <numeric> // std::partial_sum

#include "base/randgen.h"

namespace particlesystem {

void initParameters(SimParameters &params)
{
	// TODO: Validate that variables are within expected values.
	params.radius_sqr = params.radius * params.radius;
	params.sqrt_DTD = sqrt(params.diffusion * params.timestep);

	double sumC = 0.0;
	for (auto c : params.conc) { sumC += c; }

	if (sumC == 0.0)
	{
		std::cerr << "[ERROR] The total concentration is null." << std::endl;
		exit(EXIT_FAILURE);
	}

	for (auto& c : params.conc) { c /= sumC; }

	const char *abc = {"ABCDEFGHIJKLMNOPQRSTUVWXYZ"};

	// Check for identical concentrations
	for (int i = 0; i < NUM_TYPES; i++)
	{
		for (int j = i+1; j < NUM_TYPES; j++)
		{
			if ( (params.conc[i] * params.conc[j] > 0) &&
				 (params.alpha[i] == params.alpha[j]) &&
				 (params.mu[i] == params.mu[j]) )
			{
				std::cerr << "[ERROR] Identical concentrations: "
						  << abc[i] << abc[j] << std::endl;
				exit(EXIT_FAILURE);
			}
		}
	}

	std::cout << "[INFO] Concentrations:";
	for (auto c : params.conc) { std::cout << " " << c; }
	std::cout << std::endl;
}

template <typename T>
void reservoirSample(T *dst, const T *src, int ndst, int nsrc, RandGen<T> *rng)
{
	int i, j;
	for (i = 0; i < ndst; i++)
	{
		dst[2*i] = src[2*i];
		dst[2*i+1] = src[2*i+1];
	}
	for (i = ndst; i < nsrc; i++)
	{
		j = (int)(rng->next() * i);
		if (j < ndst)
		{
			dst[2*j] = src[2*i];
			dst[2*j+1] = src[2*i+1];
		}
	}
}

template <typename T>
void assignAlphaMu(T *alpha_mu, const T *conc, const T *alpha, const T *mu,
                   unsigned int arr_size, RandGen<T> *rng)
{
	T partialSum[NUM_TYPES];
	std::partial_sum(conc, conc + NUM_TYPES, partialSum);

	for (unsigned int i = 0; i < arr_size; i++)
	{
		T random = rng->next();
		for (int j = 0; j < NUM_TYPES; j++)
		{
			if (random <= partialSum[j])
			{
				alpha_mu[2*i] = alpha[j];
				alpha_mu[2*i+1] = mu[j];
				break;
			}
		}
	}
}

void initParticles(double* positions, double* alpha_mu, SimParameters params,
                   unsigned long long seed)
{
	double noisePos = 0.01;
	double boxLength = params.boxlength;
	unsigned int numParticles = params.num_elements;

	double density = numParticles / (boxLength * boxLength);
	double a = sqrt(2 / density / sqrt(3));
	int nx = (int)ceil(boxLength / a);
	int ny = (int)ceil(boxLength / (a * sqrt(3)));

	double dx = boxLength / nx;
	double dy = boxLength / ny;
	std::cerr << "# a: " << a << " Dx: " << dx << " Dy " << dy
			  << " Nx " << nx << " Ny " << ny << std::endl;

	int nTemp = 2 * nx * ny;

	// Create RNG with exactly the number of random values that will be used.
	auto *rng = new RandGen<double>(3 * nTemp, seed);
	double *tempPos = new double[2 * nTemp];

	int j = 0;
	for (int k = 0; k < nx; k++)
	{
		for (int l = 0; l < ny; l++)
		{
			tempPos[2*j] = dx * (k + 0.25) + noisePos * (rng->next() - 0.5);
			tempPos[2*j+1] = dy * (l + 0.25) + noisePos * (rng->next() - 0.5);
			j++;
			tempPos[2*j] = dx * (k + 0.75) + noisePos * (rng->next() - 0.5);
			tempPos[2*j+1] = dy * (l + 0.75) + noisePos * (rng->next() - 0.5);
			j++;
		}
	}
	std::cerr << "#NDiscos " << numParticles << " j " << j << std::endl;

	reservoirSample<double>(positions, tempPos, numParticles, nTemp, rng);
	assignAlphaMu<double>(alpha_mu, params.conc, params.alpha, params.mu,
			              params.num_elements, rng);

	delete [] tempPos;
	delete rng;
}

} // namespace particlesystem
