#include "particlesystem/particlesystem_verlet.h"

#include <cassert>
#include <cstdio>

#include "base/cuda_check.h"
#include "base/kernel_util.h"
#include "base/kernel_random.cuh"
#include "base/math_functions.cuh"
#include "base/simparameters.h"

#define EPSILON_DISTANCE		0.001
#define DISTANCE_SOFTENING		1e-9

namespace particlesystem {
namespace verlet {

__constant__ SimParameters consts;
__constant__ VerletParameters vconsts;

void ParticleSystemVerlet::initCuda(unsigned long long seed)
{
	// Load simulation data
	cudaCheck(cudaMemcpyToSymbol(consts, &params_, sizeof(SimParameters)));
	cudaCheck(cudaMemcpyToSymbol(vconsts, &verlet_, sizeof(VerletParameters)));

	// Initialize rng structure
	cudaCheck(cudaMalloc(&devicedata_.rng_states,
			             params_.num_elements * sizeof(curandState)));
	setupRng(devicedata_.rng_states, seed, params_.num_elements,
	         hParticleLaunch);
}

/* [15 FLOPS]
 * Computes the velocity of particle i by effect of particle j.
 */
__device__ __host__ double2
two_body_interaction(double2 vi, double2 pi, double2 ti, double2 pj, double2 tj,
                     double boxLength)
{
	// r_ij [5 FLOPS]
	double2 r = distVec(pi, pj, boxLength);

	// distSqr = dot(r_ij, r_ij) [3 FLOPS]
	double distSqr = dot(r, r) + DISTANCE_SOFTENING;

	double dist = sqrt(distSqr);
	double invDist7 = 1 / (dist * distSqr * distSqr * distSqr);

	// s = mu_i * alpha_j * s [2 FLOPS]
	double s = tj.x * ti.y * invDist7; // ti.y = mu_i, tj.x = alpha_j

	assert(isfinite(pi.x) && isfinite(pi.y));
	assert(isfinite(pj.x) && isfinite(pj.y));
	assert(isfinite(s));
	assert(dist > 0.9);

	// v_i = v_i + s * r_ij [4 FLOPS]
	vi.x -= r.x * s;
	vi.y -= r.y * s;

	return vi;
}

__device__ __host__ double2
integrate(double2 pos, double2 vel, double2 noise, double dt, double root,
               double boxLength)
{
	pos.x += dt * vel.x + root * noise.x;
	pos.y += dt * vel.y + root * noise.y;
	pos = wrapBoundary(pos, boxLength);

	assert(isfinite(pos.x) && isfinite(pos.y));

	return pos;
}

////////////////////////////////////////////////////////////////////////////////
/// NBODY ALGORITHM - SHARED MEMORY
////////////////////////////////////////////////////////////////////////////////

__global__ void
integrateKernel(__restrict__ double2* new_pos,
			    __restrict__ const double2* old_pos,
			    __restrict__ const double2* alpha_mu,
			    const int* neighbor_counters, const int* neighbor_list,
				curandState* rng_states)
{
    unsigned int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    for(; tidx < consts.num_elements; tidx += stride)
	{
		double2 pos = old_pos[tidx];
		double2 type = alpha_mu[tidx];
		double2 vel = make_double2(0.0, 0.0);

		for (int i = 0; i < neighbor_counters[tidx]; i++)
		{
			unsigned int neighbor_idx = neighbor_list[tidx * vconsts.max_neighbors + i];
			double2 nPos = old_pos[neighbor_idx];
			double2 nType = alpha_mu[neighbor_idx];
			vel = two_body_interaction(vel, pos, type, nPos, nType,
					                   consts.boxlength);
		}

		double2 noise = gaussianNoise(rng_states, tidx);
		new_pos[tidx] = integrate(pos, vel, noise, consts.timestep,
				                       consts.sqrt_DTD, consts.boxlength);
	}
}

void ParticleSystemVerlet::integrate()
{
	integrateKernel <<< hParticleLaunch.x, hParticleLaunch.y >>>
	(devicedata_.positions[1 - current_read],
	 devicedata_.positions[current_read],
	 devicedata_.charges,
	 devicedata_.neighbor_counters,
	 devicedata_.neighbor_list,
	 devicedata_.rng_states);
	cudaCheck(cudaDeviceSynchronize());

	std::swap(current_read, current_write);
}

////////////////////////////////////////////////////////////////////////////////
/// COMPUTE OVERLAP CORRECTION FUNCTIONS
////////////////////////////////////////////////////////////////////////////////

__device__ double2
two_body_overlap(double2 disp,   // Acummulated displacement over bi
				 double2 bi,     // Position of particle bi
				 double2 bj,     // Position of particle bj
				 int *readyflag) // Global overlap counter
{
	double2 r = distVec(bi, bj, consts.boxlength);
	double dist_sqr = dot(r, r);

	if (dist_sqr < consts.radius_sqr - EPSILON_DISTANCE)
	{
		readyflag[0] = 0;

		double dist = sqrt(dist_sqr);
		double delta = consts.radius - dist;
		disp.x -= delta * r.x / dist;
		disp.y -= delta * r.y / dist;
	}

	return disp;
}

__global__ void
compute_overlaps_kernel(__restrict__ double2* new_pos,
						__restrict__ const double2* old_pos,
						int* neighbor_counters, int* neighbor_list,
						int* readyflag)
{
	const double max_disp = 0.25 * consts.radius;

	unsigned int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    for(; tidx < consts.num_elements; tidx += stride)
	{
    	double2 disp = make_double2(0.0, 0.0);
		double2 pos = old_pos[tidx];

		for (int i = 0; i < neighbor_counters[tidx]; i++)
		{
			unsigned int neighbor_idx = neighbor_list[tidx * vconsts.max_neighbors + i];
			double2 nPos = old_pos[neighbor_idx];
			disp = two_body_overlap(disp, pos, nPos, readyflag);
		}

		pos.x += clamp(disp.x, -max_disp, max_disp);
		pos.y += clamp(disp.y, -max_disp, max_disp);
		pos = wrapBoundary(pos, consts.boxlength);
		new_pos[tidx] = pos;
	}
}

void ParticleSystemVerlet::correctOverlaps()
{
	int iter = 0;
	readyflag[0] = 0;

	while (!readyflag[0])
	{
		if (iter == 1000)
			handle_error("[ERROR] Overlap correction has iterated too much on "
					     "a single time step.");

		readyflag[0] = 1;

		compute_overlaps_kernel<<< hParticleLaunch.x, hParticleLaunch.y >>>
		(devicedata_.positions[1 - current_read],
		 devicedata_.positions[current_read],
		 devicedata_.neighbor_counters,
		 devicedata_.neighbor_list,
		 devicedata_.readyflag);
		cudaCheck(cudaDeviceSynchronize());

		std::swap(current_read, current_write);

		iter++;
	}
	overlap_counter += iter;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/* Compares positions at the current time step with the positions at the moment
 * the verlet List was constructed, marking the list as obsolete if any particle
 * has moved long enough.
 */
__global__ void
decide_to_build_verlet_kernel(const double2* new_pos, const double2* old_pos,
							  int* rebuildFlag)
{
    unsigned int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    for(; tidx < consts.num_elements; tidx += stride)
	{
		double2 pos = new_pos[tidx];
		double2 oldpos = old_pos[tidx];

		double2 r = distVec(pos, oldpos, consts.boxlength);
		double dist_sqr = dot(r, r);
		if (dist_sqr > vconsts.deltaskin_sqr)
		{
			rebuildFlag[0] = 1;	// rebuild verlet list
		}
	}
}

void ParticleSystemVerlet::verifyVerletList()
{
	readyflag[0] = 0;

	decide_to_build_verlet_kernel<<< hParticleLaunch.x, hParticleLaunch.y >>>
	(devicedata_.positions[current_read],
	 devicedata_.positions[2],
	 devicedata_.readyflag);
	cudaCheck(cudaDeviceSynchronize());

	if (readyflag[0])
	{
		//std::cout << "Rebuilding verlet" << std::endl;
		buildVerletList();
	}
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

__device__ uint2 calc_grid_pos(double2 pos)
{
	uint2 gridpos; // cast = floor for positives
	gridpos.x = (unsigned int) (pos.x / vconsts.cell_length);
	gridpos.y = (unsigned int) (pos.y / vconsts.cell_length);
	return gridpos;
}

__device__ unsigned int calc_grid_hash(uint2 gridpos)
{
	return gridpos.x * vconsts.grid_size + gridpos.y;
}

__global__ void
build_cell_list_kernel(const double2* positions, double2* aux_pos,
					   int* cell_counters, int* cell_list)
{
    unsigned int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    for(; tidx < consts.num_elements; tidx += stride)
	{
		double2 pos = positions[tidx];
		aux_pos[tidx] = pos;

		uint2 gridpos = calc_grid_pos(pos);
		unsigned int hash = calc_grid_hash(gridpos);

		assert(hash < vconsts.grid_size * vconsts.grid_size);

		if (cell_counters[hash] < vconsts.cell_size)
		{
			// atomic addition returns cellCounters[n] + 1
	 		int counter = atomicAdd(&cell_counters[hash], 1);
			cell_list[counter + hash * vconsts.cell_size] = tidx;
		}
		else printf("Reached cell limit in cell %u\n", hash);
	}
}



/* Builds the verlet list, using lists of the particles that belong to each
 * cell of the grid and lists of neighbors of each particle.
 */
__global__ void
build_verlet_list_kernel(const double2* positions,
						 int* neighbor_counters, int* neighbor_list,
						 const int* cell_counters, const int* cell_list)
{
    unsigned int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    for(; tidx < consts.num_elements; tidx += stride)
	{
    	double2 pos = positions[tidx];
    	uint2 gridpos = calc_grid_pos(pos);

		for (int dnx = -1; dnx <= 1; dnx++)
		{
			for (int dny = -1; dny <= 1; dny++)
			{
				unsigned int nxv = (gridpos.x + dnx + vconsts.grid_size) % vconsts.grid_size;
				unsigned int nyv = (gridpos.y + dny + vconsts.grid_size) % vconsts.grid_size;
				unsigned int nv = nxv * vconsts.grid_size + nyv;

				for (unsigned int j = 0; j < cell_counters[nv]; j++)
				{
					unsigned int ntid = cell_list[nv * vconsts.cell_size + j];

					assert(ntid < consts.num_elements);

					if (tidx != ntid) // Avoid comparing the particle with itself
					{
						double2 r = distVec(pos, positions[ntid],
								                    consts.boxlength);

						double dist_sqr = dot(r, r);
						if (dist_sqr < vconsts.skin_sqr)
						{
							if (neighbor_counters[tidx] < vconsts.max_neighbors)
							{
								int counter = atomicAdd(&neighbor_counters[tidx], 1);
								neighbor_list[tidx * vconsts.max_neighbors + counter] = ntid;
							}
							else printf("Reached neighbor limit in particle %u\n", tidx);
						}
					}
				}
			}
		}
	}
}

void ParticleSystemVerlet::buildVerletList()
{
	memsetArray<int>(devicedata_.neighbor_counters, 0, params_.num_elements,
			         hParticleLaunch);
	memsetArray<int>(devicedata_.cell_counters, 0,
			         verlet_.grid_size * verlet_.grid_size, hParticleLaunch);

	build_cell_list_kernel<<< hParticleLaunch.x, hParticleLaunch.y >>>
	(devicedata_.positions[current_read],
	 devicedata_.positions[2],
	 devicedata_.cell_counters,
	 devicedata_.cell_list);
	cudaCheck(cudaDeviceSynchronize());

	build_verlet_list_kernel<<< hParticleLaunch.x, hParticleLaunch.y >>>
	(devicedata_.positions[current_read],
	 devicedata_.neighbor_counters,
	 devicedata_.neighbor_list,
	 devicedata_.cell_counters,
	 devicedata_.cell_list);
	cudaCheck(cudaDeviceSynchronize());
}


__global__ void
calc_hash_kernel(unsigned int* grid_particle_hash,
				 unsigned int* grid_particle_idx,
				 const double2* positions)
{
	unsigned int tidx = blockIdx.x * blockDim.x + threadIdx.x;

	if (tidx < consts.num_elements)
	{
		volatile double2 pos = positions[tidx];
		uint2 gridpos = calc_grid_pos(pos);
		unsigned int hash = calc_grid_hash(gridpos);

		grid_particle_hash[tidx] = hash;
		grid_particle_idx[tidx] = tidx;
	}
}

__global__ void
reorder_data_kernel(unsigned int* cell_start, unsigned int* cell_end,
					double2 *sorted_pos,
					unsigned int* grid_particle_hash,
					unsigned int* grid_particle_idx,
					double2* old_pos)
{
	extern __shared__ unsigned int sh_hash[];
	unsigned int tidx = blockIdx.x * blockDim.x + threadIdx.x;

	unsigned int hash;

	if (tidx < consts.num_elements)
	{
		hash = grid_particle_hash[tidx];
		sh_hash[threadIdx.x + 1] = hash;

		if (tidx > 0 && threadIdx.x == 0)
		{
			sh_hash[0] = grid_particle_hash[tidx - 1];
		}
	}

	__syncthreads();

	if (tidx < consts.num_elements)
	{
		if (tidx == 0 || hash != sh_hash[threadIdx.x])
		{
			cell_start[hash] = tidx;
			if (tidx > 0) { cell_end[sh_hash[threadIdx.x]] = tidx; }
		}

		if (tidx == consts.num_elements - 1) { cell_end[hash] = tidx + 1; }
	}

	unsigned int sortedIndex = grid_particle_idx[tidx];
	double2 pos = old_pos[sortedIndex];
	sorted_pos[tidx] = pos;
}

} // namespace verlet
} // namespace particlesystem
