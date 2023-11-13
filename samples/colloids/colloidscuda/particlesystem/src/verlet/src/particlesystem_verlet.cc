#include "particlesystem/particlesystem_verlet.h"

#include <math_constants.h> // CUDART_PI

#include <cstdio>

#include "base/cuda_check.h"

namespace particlesystem {
namespace verlet {

ParticleSystemVerlet::ParticleSystemVerlet(SimParameters p, VerletParameters vp)
	: ParticleSystemVerlet(p, vp, time(nullptr))
{}

ParticleSystemVerlet::ParticleSystemVerlet(SimParameters p, VerletParameters vp,
                                           unsigned long long seed):
	params_(p),
	verlet_(vp),
	current_read(0),
	current_write(1),
	overlap_counter(0),
	readyflag(nullptr)
{
	initParameters(params_);

	positions_ = new double[params_.num_elements * 2];
	charges_ = new double[params_.num_elements * 2];
	hParticleCounter = new int[params_.num_elements];

	initParticles(positions_, charges_, params_, seed);
	initCommon();
	initCuda(seed);
	loadOnDevice();
	buildVerletList();
}

ParticleSystemVerlet::ParticleSystemVerlet(SimParameters p, VerletParameters vp,
                                           unsigned long long seed,
                                           std::string pFileName):
	params_(p),
	verlet_(vp),
	current_read(0),
	current_write(1),
	overlap_counter(0),
	readyflag(nullptr)
{
	initParameters(params_);

	positions_ = new double[params_.num_elements * 2];
	charges_ = new double[params_.num_elements * 2];
	hParticleCounter = new int[params_.num_elements];

	readFile(pFileName);
	initCommon();
	initCuda(seed);
	loadOnDevice();
	buildVerletList();
}

ParticleSystemVerlet::~ParticleSystemVerlet()
{
	finalize();
}

void ParticleSystemVerlet::runSimulation(int num_iter,
                                         unsigned int save_period=0)
{
	int iter;
	float time = 0.0f;

	cudaEvent_t start, stop;
	cudaCheck(cudaEventCreate(&start, cudaEventDefault));
	cudaCheck(cudaEventCreate(&stop, cudaEventDefault));
	cudaCheck(cudaEventRecord(start, 0));

	for (iter = 1; iter <= num_iter; iter++)
	{
		// Print the current iteration, and sync data with host
		if (save_period != 0 && iter % save_period == 0)
		{
			printf("Iter = %d; t = %f\n", iter, iter * params_.timestep);
			save_config();
		}

		runTimestep();
	}
	cudaCheck(cudaEventRecord(stop, 0));
	cudaCheck(cudaEventSynchronize(stop));
	cudaCheck(cudaEventElapsedTime(&time, start, stop));
	cudaCheck(cudaEventDestroy(start));
	cudaCheck(cudaEventDestroy(stop));

	printf("Average simulation time (ms): %f\n"
		   "Total simulation time (ms): %f\n"
		   "Average overlap correction iterations: %f\n",
		   time / iter, time, static_cast<float>(overlap_counter) / time);
}

void ParticleSystemVerlet::runTimestep()
{
	integrate();
	correctOverlaps();
	verifyVerletList();
}

void ParticleSystemVerlet::initCommon()
{
	double particleArea = pow(params_.radius / 2, 2) * CUDART_PI;
	double boxArea = params_.boxlength * params_.boxlength;
	double packing = (params_.num_elements * particleArea) / boxArea;

	printf("Simulation parameters:\n"
		   "Particle count: %u\n"
		   "Packing fraction: %lf\n"
		   "Box length: %lf\n"
		   "Cell length: %lf\n"
		   "Grid size: %u\n",
		   params_.num_elements, packing, params_.boxlength,
		   verlet_.cell_length, verlet_.grid_size);

	cudaDeviceProp deviceProp;
	cudaCheck(cudaSetDevice(0));
	cudaCheck(cudaGetDeviceProperties(&deviceProp, 0));

	unsigned int blockSize = BLOCK_SIZE_PARTICLES;
	hParticleLaunch = make_uint2(deviceProp.multiProcessorCount, blockSize);

	printf("Setting kernel parameters (grid size, block size) = (%u, %u)\n",
	       hParticleLaunch.x, hParticleLaunch.y);
}

void ParticleSystemVerlet::loadOnDevice()
{
	// Load particle data
	size_t particlesByteSize = params_.num_elements * 2 * sizeof(double);
	cudaCheck(cudaMalloc(&devicedata_.positions[0], particlesByteSize));
	cudaCheck(cudaMalloc(&devicedata_.positions[1], particlesByteSize));
	cudaCheck(cudaMalloc(&devicedata_.positions[2], particlesByteSize));
	cudaCheck(cudaMemcpy(devicedata_.positions[current_read], positions_,
			             particlesByteSize, cudaMemcpyHostToDevice));

	// Load particle type data
	cudaCheck(cudaMalloc(&devicedata_.charges, particlesByteSize));
	cudaCheck(cudaMemcpy(devicedata_.charges, charges_, particlesByteSize,
			             cudaMemcpyHostToDevice));

	size_t neighborsByteSize = params_.num_elements * sizeof(unsigned int);
	cudaCheck(cudaMalloc(&devicedata_.neighbor_counters, neighborsByteSize));

	size_t neighborListSize = neighborsByteSize * verlet_.max_neighbors;
	cudaCheck(cudaMalloc(&devicedata_.neighbor_list, neighborListSize));

	size_t cellsByteSize = verlet_.grid_size * verlet_.grid_size * sizeof(unsigned int);
	cudaCheck(cudaMalloc(&devicedata_.cell_counters, cellsByteSize));

	size_t cellListSize = cellsByteSize * verlet_.cell_size;
	cudaCheck(cudaMalloc(&devicedata_.cell_list, cellListSize));

	// Zero-copy auxiliary variable for edge-flip algorithm
	cudaCheck(cudaHostAlloc(&readyflag, sizeof(int), cudaHostAllocMapped));
	cudaCheck(cudaHostGetDevicePointer(&devicedata_.readyflag, readyflag, 0));
}

void ParticleSystemVerlet::syncWithDevice()
{
	size_t particlesByteSize = params_.num_elements * 2 * sizeof(double);
	cudaCheck(cudaMemcpy(positions_, devicedata_.positions[current_read],
			             particlesByteSize, cudaMemcpyDeviceToHost));
	// No need to synchronize particle type data, as it stays the same
	// throughout the simulation.
}

void ParticleSystemVerlet::finalize()
{
	cudaCheck(cudaFree(devicedata_.positions[0]));
	cudaCheck(cudaFree(devicedata_.positions[1]));
	cudaCheck(cudaFree(devicedata_.positions[2]));
	cudaCheck(cudaFree(devicedata_.rng_states));

	cudaCheck(cudaFree(devicedata_.neighbor_counters));
	cudaCheck(cudaFree(devicedata_.neighbor_list));
	cudaCheck(cudaFree(devicedata_.cell_counters));
	cudaCheck(cudaFree(devicedata_.cell_list));

	cudaCheck(cudaFreeHost(readyflag));

	cudaDeviceReset();

	delete [] positions_;
	delete [] charges_;
	delete [] hParticleCounter;
}

} // namespace verlet
} // namespace particlesystem
