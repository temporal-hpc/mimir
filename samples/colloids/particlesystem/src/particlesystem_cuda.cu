
#include "particlesystem_delaunay.h"

#include <cassert>
#include <cstdio>

#include "cuda_check.h"
#include "kernel_random.cuh"
#include "kernel_util.h"
#include "math_functions.cuh"
#include "simparameters.h"

#define EPSILON_DISTANCE    0.001
#define EPSILON_ANGLE 	    0.001
#define DISTANCE_SOFTENING  1e-9

// TODO: Join common code using templates
// TODO: Decouple kernel wrappers from class

namespace particlesystem {
namespace delaunay {

__constant__ SimParameters consts;

void ParticleSystemDelaunay::initCuda(unsigned long long seed)
{
	// Load simulation data
	cudaCheck(cudaMemcpyToSymbol(consts, &params_, sizeof(SimParameters)));

	// Initialize rng structure
	cudaCheck(cudaMalloc(&devicedata_.rng_states,
			             params_.num_elements * sizeof(curandState)));
	setupRng(devicedata_.rng_states, seed, params_.num_elements,
	         hParticleLaunch);
}

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double
atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull,
        				assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

//! 39 flop
__device__ __host__ int
delaunayTest2d(const double2* particles, const int op1, const int op2,
			   const int com_a, const int com_b, const double boxlength)
{
	double2 u; // vector
	double2 p, q; // points
	// get two vectors of the first triangle
	p = particles[op1];
	q = particles[com_a];
	u = distVec(p, q, boxlength); //! + 5 flop
	q = particles[com_b];
	double alpha = angle(u, distVec(p, q, boxlength)); //! + 11 flop
	// the same for other triangle
	p = particles[op2];
	q = particles[com_a];
	u = distVec(p, q, boxlength); //! + 5 flop
	q = particles[com_b];
	double beta = angle(u, distVec(p, q, boxlength)); //! + 11 flop

	return (int)(abs(alpha + beta) / (180.0 + EPSILON_ANGLE)); //! + 7 flop
}

__device__ __host__ int
invertedTriangleTest(double2 op1, double2 op2, double2 e1, double2 e2,
                     double boxlength)
{
	double2 v0 = distVec(e1, e2, boxlength);
	double2 v1 = distVec(e1, op1, boxlength);
	double2 v2 = distVec(e1, op2, boxlength);

	double d = cross(v2, v0);
	double s = cross(v1, v0);
	double t = cross(v2, v1);

    return (d < 0 && s <= 0 && t <= 0 && s+t >= d) ||
    	   (d > 0 && s >= 0 && t >= 0 && s+t <= d) ||
    	   (s < 0 && d <= 0 && -t <= 0 && d-t >= s) ||
    	   (s > 0 && d >= 0 && -t >= 0 && d-t <= s);
}

////////////////////////////////////////////////////////////////////////////////
/// Kernel -- Copy arrays
////////////////////////////////////////////////////////////////////////////////

template<unsigned int block_size>
__global__ void
correctTrianglesKernel(double2* positions, unsigned int* triangles,
                       int2* edge_idx, int2* edge_ta, int2* edge_tb,
                       int2* edge_op, int* readyflag, int* rotations,
                       int* reserved, unsigned int num_edges)
{
    __shared__ int2 shEdgesTa[block_size];
    __shared__ int2 shEdgesTb[block_size];
    __shared__ int2 shOpposites[block_size];

	unsigned int tidx = blockIdx.x * blockDim.x + threadIdx.x; //! + 2 flop
    if (tidx < num_edges)
	{
		shEdgesTa[threadIdx.x] = edge_ta[tidx];
		shEdgesTb[threadIdx.x] = edge_tb[tidx];
		shOpposites[threadIdx.x] = edge_op[tidx];

		__syncthreads();

		if (shEdgesTb[threadIdx.x].x != -1)
		{
			if (invertedTriangleTest(positions[triangles[shOpposites[threadIdx.x].x]],
					           	   	 positions[triangles[shOpposites[threadIdx.x].y]],
					                 positions[triangles[shEdgesTa[threadIdx.x].x]],
					                 positions[triangles[shEdgesTa[threadIdx.x].y]],
					 	 	 	 	 consts.boxlength) > 0)
			{
				readyflag[0] = 0;
				// exclusion part
				if (atomicExch(&(reserved[shEdgesTa[threadIdx.x].y/3]), tidx) == -1 &&
					atomicExch(&(reserved[shEdgesTb[threadIdx.x].y/3]), tidx) == -1)
				{ //!  + 8 flop

					// proceed to flip the edges
					rotations[shEdgesTa[threadIdx.x].y/3] = shEdgesTb[threadIdx.x].y/3; //! + 8 flop
					rotations[shEdgesTb[threadIdx.x].y/3] = shEdgesTa[threadIdx.x].y/3; //! + 8 flop
					// exchange necessary indexes
					triangles[shEdgesTa[threadIdx.x].x] = triangles[shOpposites[threadIdx.x].y];
					triangles[shEdgesTb[threadIdx.x].y] = triangles[shOpposites[threadIdx.x].x];
					// update the indices of the flipped edge.
					edge_ta[tidx] = make_int2(shOpposites[threadIdx.x].x, shEdgesTa[threadIdx.x].x);
					edge_tb[tidx] = make_int2(shEdgesTb[threadIdx.x].y, shOpposites[threadIdx.x].y);
					// update vertex indices
					edge_idx[tidx] = make_int2(triangles[shOpposites[threadIdx.x].x], triangles[shEdgesTa[threadIdx.x].x]);
					// update oppposites indices
					edge_op[tidx] = make_int2(shEdgesTa[threadIdx.x].y, shEdgesTb[threadIdx.x].x);
				}
			}
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
/// CLEAP::KERNEL:: delaunay transformation :: exclussion & processing 2D
////////////////////////////////////////////////////////////////////////////////

//! 2D --> 65 flop
template<unsigned int block_size>
__global__ void
exclusionProcessingKernel(double2* positions, unsigned int* triangles,
						  int2* edge_idx, int2* edge_ta, int2* edge_tb,
						  int2* edge_op, int* readyflag, int* rotations,
						  int* locks, unsigned int num_edges)
{
    __shared__ int2 shEdgesTa[block_size];
    __shared__ int2 shEdgesTb[block_size];
    __shared__ int2 shOpposites[block_size];

	unsigned int tidx = blockIdx.x * blockDim.x + threadIdx.x; //! + 2 flop
    if (tidx < num_edges)
	{
		shEdgesTa[threadIdx.x] = edge_ta[tidx];
		shEdgesTb[threadIdx.x] = edge_tb[tidx];
		shOpposites[threadIdx.x] = edge_op[tidx];

		//__syncthreads();

		if (shEdgesTb[threadIdx.x].x != -1)
		{
			if (delaunayTest2d(positions,
							   triangles[shOpposites[threadIdx.x].x],
							   triangles[shOpposites[threadIdx.x].y],
							   triangles[shEdgesTa[threadIdx.x].x],
							   triangles[shEdgesTa[threadIdx.x].y],
							   consts.boxlength) > 0)
			{
				readyflag[0] = 0;
				// exclusion part
				if (atomicExch(&(locks[shEdgesTa[threadIdx.x].y/3]), tidx) == -1 &&
					atomicExch(&(locks[shEdgesTb[threadIdx.x].y/3]), tidx) == -1)
				{ //!  + 8 flop
					assert(triangles[shEdgesTa[threadIdx.x].x] == triangles[shEdgesTb[threadIdx.x].x]
					    && triangles[shEdgesTa[threadIdx.x].y] == triangles[shEdgesTb[threadIdx.x].y]);

					// proceed to flip the edges
					rotations[shEdgesTa[threadIdx.x].y/3] = shEdgesTb[threadIdx.x].y/3; //! + 8 flop
					rotations[shEdgesTb[threadIdx.x].y/3] = shEdgesTa[threadIdx.x].y/3; //! + 8 flop
					// exchange necessary indexes
					triangles[shEdgesTa[threadIdx.x].x] = triangles[shOpposites[threadIdx.x].y];
					triangles[shEdgesTb[threadIdx.x].y] = triangles[shOpposites[threadIdx.x].x];
					// update the indices of the flipped edge.
					edge_ta[tidx] = make_int2(shOpposites[threadIdx.x].x, shEdgesTa[threadIdx.x].x);
					edge_tb[tidx] = make_int2(shEdgesTb[threadIdx.x].y, shOpposites[threadIdx.x].y);
					// update vertex indices
					edge_idx[tidx] = make_int2(triangles[shOpposites[threadIdx.x].x], triangles[shEdgesTa[threadIdx.x].x]);
					// update oppposites indices
					edge_op[tidx] = make_int2(shEdgesTa[threadIdx.x].y, shEdgesTb[threadIdx.x].x);

					assert(edge_idx[tidx].x != edge_idx[tidx].y);
				}
			}
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
/// CLEAP::KERNEL:: delaunay transformation :: Repair
////////////////////////////////////////////////////////////////////////////////

__global__ void
repairTrianglesKernel(unsigned int* triangles, int* rotations,
					  int2* edge_idx, int2* edge_ta, int2* edge_tb, int2* edge_op,
					  unsigned int num_edges)
{
    unsigned int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tidx < num_edges)
    {
	// use volatile variables, this forces register use. Sometimes manual optimization achieves better performance.
        volatile int2 e = edge_idx[tidx];
        volatile int2 ta = edge_ta[tidx];
        volatile int2 tb = edge_tb[tidx];
        volatile int2 op = edge_op[tidx];
	// if the t_a pair of indexes are broken
        if ((e.x != triangles[ta.x] || e.y != triangles[ta.y]))
        {
	    // then repair them.
            int t_index = rotations[ ta.x/3 ];
            if (triangles[3*t_index+0] == e.x)
            {
               ta.x = 3*t_index+0;
               triangles[3*t_index+1] == e.y ? (ta.y = 3*t_index+1, op.x = 3*t_index+2) : (ta.y = 3*t_index+2, op.x = 3*t_index+1);
            }
            else if (triangles[3*t_index+1] == e.x)
            {
               ta.x = 3*t_index+1;
               triangles[3*t_index+0] == e.y ? (ta.y = 3*t_index+0, op.x = 3*t_index+2) : (ta.y = 3*t_index+2, op.x = 3*t_index+0);
            }
            else if (triangles[3*t_index+2] == e.x)
            {
               ta.x = 3*t_index+2;
               triangles[3*t_index+0] == e.y ? (ta.y = 3*t_index+0, op.x = 3*t_index+1) : (ta.y = 3*t_index+1, op.x = 3*t_index+0);
            }
        }
        if (tb.x != -1)
        {
            if ((e.x != triangles[tb.x] || e.y != triangles[tb.y]))
            {
                int t_index = rotations[ tb.x/3 ];
                if (triangles[3*t_index+0] == e.x)
                {
                   tb.x = 3*t_index+0;
                   triangles[3*t_index+1] == e.y ? (tb.y = 3*t_index+1, op.y = 3*t_index+2) : (tb.y = 3*t_index+2, op.y = 3*t_index+1);
                }
                else if(triangles[3*t_index+1] == e.x)
                {
                   tb.x = 3*t_index+1;
                   triangles[3*t_index+0] == e.y ? (tb.y = 3*t_index+0, op.y = 3*t_index+2) : (tb.y = 3*t_index+2, op.y = 3*t_index+0);
                }
                else if(triangles[3*t_index+2] == e.x)
                {
                   tb.x = 3*t_index+2;
                   triangles[3*t_index+0] == e.y ? (tb.y = 3*t_index+0, op.y = 3*t_index+1) : (tb.y = 3*t_index+1, op.y = 3*t_index+0);
                }
            }
        }
        edge_ta[tidx] = make_int2(ta.x, ta.y);
        edge_tb[tidx] = make_int2(tb.x, tb.y);
        edge_op[tidx] = make_int2(op.x, op.y);
    }
}

void ParticleSystemDelaunay::updateDelaunay()
{
	int iter = 0;
	readyflag[0] = 0;

	while (!readyflag[0])
	{
		if (iter == 1000)
			handleError("[ERROR] Delaunay updating has iterated too much on a "
					     "single time step.");

		readyflag[0] = 1;
		memsetDualArrays<int>(devicedata_.dTriRel, devicedata_.dTriReserv, -1,
				              delaunay_.num_triangles, hTriangleLaunch);

		exclusionProcessingKernel<BLOCK_SIZE_EDGES> <<< hEdgeLaunch.x, hEdgeLaunch.y >>>
		(devicedata_.positions[current_read],
		 devicedata_.triangles,
		 devicedata_.edge_idx,
		 devicedata_.edge_ta,
		 devicedata_.edge_tb,
		 devicedata_.edge_op,
		 devicedata_.readyflag,
		 devicedata_.dTriRel,
		 devicedata_.dTriReserv,
		 delaunay_.num_edges);
		cudaCheck(cudaDeviceSynchronize());

		if (readyflag[0]) { break; }

		repairTrianglesKernel<<< hEdgeLaunch.x, hEdgeLaunch.y >>>
		(devicedata_.triangles,
		 devicedata_.dTriRel,
		 devicedata_.edge_idx,
		 devicedata_.edge_ta,
		 devicedata_.edge_tb,
		 devicedata_.edge_op,
		 delaunay_.num_edges);
		cudaCheck(cudaDeviceSynchronize());

		iter++;
	}
	flip_counter += iter;
}

void ParticleSystemDelaunay::updateTriangles()
{
	int iter = 0;
	readyflag[0] = 0;

	while (!readyflag[0])
	{
		if (iter == 1000)
			handleError("[ERROR] Inverted triangle correction has iterated too "
					     "much on a single time step.");

		readyflag[0] = 1;
		memsetDualArrays<int>(devicedata_.dTriRel, devicedata_.dTriReserv, -1,
				              delaunay_.num_triangles, hTriangleLaunch);

		correctTrianglesKernel<BLOCK_SIZE_EDGES> <<< hEdgeLaunch.x, hEdgeLaunch.y >>>
		(devicedata_.positions[current_read],
		 //devicedata_.positions[1 - current_read],
		 devicedata_.triangles,
		 devicedata_.edge_idx,
		 devicedata_.edge_ta,
		 devicedata_.edge_tb,
		 devicedata_.edge_op,
		 devicedata_.readyflag,
		 devicedata_.dTriRel,
		 devicedata_.dTriReserv,
		 delaunay_.num_edges);
		cudaCheck(cudaDeviceSynchronize());

		if (readyflag[0]) { break; }

		repairTrianglesKernel<<< hEdgeLaunch.x, hEdgeLaunch.y >>>
		(devicedata_.triangles,
		 devicedata_.dTriRel,
		 devicedata_.edge_idx,
		 devicedata_.edge_ta,
		 devicedata_.edge_tb,
		 devicedata_.edge_op,
		 delaunay_.num_edges);
		cudaCheck(cudaDeviceSynchronize());

		iter++;
	}
}

////////////////////////////////////////////////////////////////////////////////
/// NBODY ALGORITHM - SHARED MEMORY
////////////////////////////////////////////////////////////////////////////////

/* [15 FLOPS]
 * Computes the velocity of particle i by effect of particle j.
 */
__device__ __host__
double2 two_body_interaction(double2 vi, double2 pi, double2 ti,
                             double2 pj, double2 tj, double boxlength)
{
	// r_ij [5 FLOPS]
	double2 r = distVec(pi, pj, boxlength);

	// distSqr = dot(r_ij, r_ij) [3 FLOPS]
	double distSqr = dot(r, r) + DISTANCE_SOFTENING;

	// invDistCube = 1 / distSqr^(3/2) [4 FLOPS]
	double invDist = rsqrt(distSqr);
	double invDistCube = invDist * invDist * invDist;

	// s = mu_i * alpha_j * s [2 FLOPS]
	double s = tj.x * ti.y * invDistCube; // ti.y = mu_i, tj.x = alpha_j

	// v_i = v_i + s * r_ij [4 FLOPS]
	vi.x -= r.x * s;
	vi.y -= r.y * s;

	return vi;
}

template<unsigned int block_size>
__device__ double2
computeForce(double2 pos, double2 type,
			 __restrict__ const double2* positions,
			 __restrict__ const double2* charges,
			 int num_tiles) // Number of tiles for the n-body algorithm
{
	__shared__ double2 sh_pos[block_size];
	__shared__ double2 sh_charge[block_size];

	double2 vel = make_double2(0.0, 0.0);

	for (int tile = 0; tile < num_tiles; tile++)
	{
		unsigned int tid = tile * blockDim.x + threadIdx.x;

		sh_pos[threadIdx.x] = (tid < consts.num_elements)? positions[tid]
		                                               : make_double2(0.0, 0.0);
		sh_charge[threadIdx.x] = (tid < consts.num_elements)? charges[tid]
		                                               : make_double2(0.0, 0.0);

		__syncthreads();

#pragma unroll block_size
		for (unsigned int counter = 0; counter < blockDim.x; counter++)
		{
			vel = two_body_interaction(vel, pos, type, sh_pos[counter],
					                   sh_charge[counter], consts.boxlength);
		}
		__syncthreads();
	}

	return vel;
}

template<unsigned int block_size>
__global__ void
integrateKernel(__restrict__ double2* new_pos,
				__restrict__ const double2* old_pos,
				__restrict__ const double2* charges,
				__restrict__ double2* velocities,
				curandState* rng_states, int num_tiles)
{
	unsigned int tidx = blockIdx.x * blockDim.x + threadIdx.x;

	double2 force;
	double2 pos = make_double2(0.0, 0.0);
	double2 charge = make_double2(0.0, 0.0);

	if (tidx < consts.num_elements)
	{
		pos = old_pos[tidx];
		charge = charges[tidx];
	}

	force = computeForce<block_size>(pos, charge, old_pos, charges, num_tiles);

	if (tidx < consts.num_elements)
	{
		// Compute velocity
		double2 noise = gaussianNoise(rng_states, tidx);
		double2 vel = make_double2(0.0, 0.0);
		vel.x = consts.timestep * force.x + consts.sqrt_DTD * noise.x;
		vel.y = consts.timestep * force.y + consts.sqrt_DTD * noise.y;

		// Integrate particle
		pos.x += vel.x;
		pos.y += vel.y;

		// Write position and velocity to global memory
		new_pos[tidx] = wrapBoundary(pos, consts.boxlength);
		velocities[tidx] = vel;
	}
}

////////////////////////////////////////////////////////////////////////////////
/// NBODY ALGORITHM - SHUFFLE INSTRUCTION
////////////////////////////////////////////////////////////////////////////////

__device__
double2 forceShuffle(double2 pos, double2 type,
					 __restrict__ const double2* positions,
					 __restrict__ const double2* charges,
					 int lane_idx)
{
	const unsigned int numThreads = blockDim.x * gridDim.x;
	double2 sh_pos, sh_charge;
	double2 vel = make_double2(0.0, 0.0);

	for (unsigned int warp_idx = 0; warp_idx < numThreads; warp_idx += warpSize)
	{
		unsigned int body_idx = warp_idx + lane_idx;

		sh_pos = make_double2(0.0, 0.0);
		sh_charge = make_double2(0.0, 0.0);
		if (body_idx < consts.num_elements)
		{
			sh_pos = positions[body_idx];
			sh_charge = charges[body_idx];
		}

		double shx = sh_pos.x;
		double shy = sh_pos.y;
		double sh_alpha = sh_charge.x;

#pragma unroll 32
		for (unsigned int lane = 0; lane < warpSize; lane++)
		{
			double2 pj;
            pj.x = __shfl_sync(0xFFFFFFFF, shx, lane, warpSize);
			pj.y = __shfl_sync(0xFFFFFFFF, shy, lane, warpSize);
			//pj.x = __shfl(shx, lane, warpSize);
			//pj.y = __shfl(shy, lane, warpSize);
			double2 r = distVec(pos, pj, consts.boxlength);

			double distsqr = dot(r, r) + DISTANCE_SOFTENING;
			double inv_dist = rsqrt(distsqr);
			double inv_dist_cube = inv_dist * inv_dist * inv_dist;

			double s = __shfl_sync(0xFFFFFFFF, sh_alpha, lane, warpSize) * type.y * inv_dist_cube;
            //double s = __shfl(sh_alpha, lane, warpSize) * type.y * inv_dist_cube;
			vel.x -= r.x * s;
			vel.y -= r.y * s;
		}
		__syncthreads();
	}

	return vel;
}

__global__ void
integrateShuffleKernel(__restrict__ double2* new_pos,
					   __restrict__ const double2* old_pos,
					   __restrict__ const double2* charges,
					   __restrict__ double2* velocities,
					   curandState* rng_states)
{
	const int lane_idx = threadIdx.x & (warpSize - 1);
	unsigned int tidx = blockIdx.x * blockDim.x + threadIdx.x;

	double2 pos = make_double2(0.0, 0.0);
	double2 charge = make_double2(0.0, 0.0);

	if (tidx < consts.num_elements)
	{
		pos = old_pos[tidx];
		charge = charges[tidx];
		double2 force = forceShuffle(pos, charge, old_pos, charges, lane_idx);

		// Compute velocity
		double2 vel;
		vel.x = consts.timestep * force.x;
		vel.y = consts.timestep * force.y;

		// Integrate particle
		double2 noise = gaussianNoise(rng_states, tidx);
		pos.x += vel.x + consts.sqrt_DTD * noise.x;
		pos.y += vel.y + consts.sqrt_DTD * noise.y;

		// Write position and velocity to global memory
		new_pos[tidx] = wrapBoundary(pos, consts.boxlength);
		velocities[tidx] = vel;
	}
}

void ParticleSystemDelaunay::integrateShuffle()
{
	integrateShuffleKernel <<< hParticleLaunch.x, hParticleLaunch.y >>>
	(devicedata_.positions[1 - current_read],
	 devicedata_.positions[current_read],
	 devicedata_.charges,
	 devicedata_.velocities,
	 devicedata_.rng_states);
	cudaCheck(cudaDeviceSynchronize());

	std::swap(current_read, current_write);
}

void ParticleSystemDelaunay::integrateShared()
{
	integrateKernel<BLOCK_SIZE_PARTICLES> <<< hParticleLaunch.x, hParticleLaunch.y >>>
	(devicedata_.positions[1 - current_read],
	 devicedata_.positions[current_read],
	 devicedata_.charges,
	 devicedata_.velocities,
	 devicedata_.rng_states,
	 hParticleLaunch.x);
	cudaCheck(cudaDeviceSynchronize());

	std::swap(current_read, current_write);
}

////////////////////////////////////////////////////////////////////////////////
/// OVERLAP CORRECTION FUNCTIONS
////////////////////////////////////////////////////////////////////////////////

__global__ void
computeOverlapsKernel(__restrict__ double2* displacements, // Global particles array with updated positions
				      __restrict__ const double2* old_pos, // Global particles array with the current positions
					  int2* edge_idx, int* readyflag, unsigned int num_edges)
{
	unsigned int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    for(; tidx < num_edges; tidx += stride)
	{
		int2 edge = edge_idx[tidx];
		double2 bi = old_pos[edge.x];
		double2 bj = old_pos[edge.y];

		double2 r = distVec(bi, bj, consts.boxlength);
		double dist_sqr = dot(r, r);

		if (dist_sqr < consts.radius_sqr - EPSILON_DISTANCE)
		{
			readyflag[0] = 0;

			double dist = sqrt(dist_sqr);
			double delta = consts.radius - dist;
			double2 disp = make_double2(delta * r.x / dist, delta * r.y / dist);

			atomicAdd(&(displacements[edge.x].x), -disp.x);
			atomicAdd(&(displacements[edge.x].y), -disp.y);

			atomicAdd(&(displacements[edge.y].x), disp.x);
			atomicAdd(&(displacements[edge.y].y), disp.y);
		}
	}
}

__global__ void
moveParticlesKernel(__restrict__ double2* positions,
					__restrict__ const double2* displacements)
{
	const double maxDisp = 0.25 * consts.radius;

    unsigned int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    for(; tidx < consts.num_elements; tidx += stride)
	{
		double2 pos = positions[tidx];
		double2 disp = displacements[tidx];

		assert(isfinite(pos.x) && isfinite(pos.y));
		assert(isfinite(disp.y) && isfinite(disp.y));

		pos.x += clamp(disp.x, -maxDisp, maxDisp);
		pos.y += clamp(disp.y, -maxDisp, maxDisp);
		pos = wrapBoundary(pos, consts.boxlength);

		positions[tidx] = pos;
	}
}

void ParticleSystemDelaunay::correctOverlaps()
{
	int iter = 0;
	readyflag[0] = 0;

	while (!readyflag[0])
	{
		if (iter == 1000)
			handleError("[ERROR] Overlap correction has iterated too much on a "
					     "single time step.");

		memsetArray<double>((double*)devicedata_.positions[1 - current_read],
				            0.0, 2 * params_.num_elements, hParticleLaunch);

		readyflag[0] = 1;

		computeOverlapsKernel<<< hEdgeLaunch.x, hEdgeLaunch.y >>>
		(devicedata_.positions[1 - current_read],
		 devicedata_.positions[current_read],
		 devicedata_.edge_idx,
		 devicedata_.readyflag,
		 delaunay_.num_edges);
		cudaCheck(cudaDeviceSynchronize());

		if (readyflag[0]) { break; }

		moveParticlesKernel <<< hParticleLaunch.x, hParticleLaunch.y >>>
		(devicedata_.positions[current_read],
		 devicedata_.positions[1 - current_read]);
		cudaCheck(cudaDeviceSynchronize());

		iter++;
	}
	overlap_counter += iter;
}

////////////////////////////////////////////////////////////////////////////////
/// TRIANGLE VALIDATION KERNEL
////////////////////////////////////////////////////////////////////////////////

__global__ void
validateTrianglesKernel(const double2* positions,
                        const unsigned int* triangles,
						int* readyflag, unsigned int num_triangles)
{
    unsigned int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    for(; tidx < num_triangles; tidx += stride)
    {
    	double2 p1 = positions[triangles[3*tidx]];
    	double2 p2 = positions[triangles[3*tidx+1]];
    	double2 p3 = positions[triangles[3*tidx+2]];

    	double2 a = distVec(p1, p2, consts.boxlength);
    	double2 b = distVec(p1, p3, consts.boxlength);

    	if (cross(a, b) < 0.0)
    	{
    		printf("(%u %u %u)\n(%lf %lf) (%lf %lf) (%lf %lf) %lf\n",
    				triangles[3*tidx], triangles[3*tidx+1], triangles[3*tidx+2],
    				p1.x, p1.y, p2.x, p2.y, p3.x, p3.y, cross(a, b));
    		readyflag[0] = 0;
    	}
    }
}

void ParticleSystemDelaunay::checkTriangulation()
{
	readyflag[0] = 1;

	validateTrianglesKernel<<< hTriangleLaunch.x, hTriangleLaunch.y >>>
	(devicedata_.positions[current_read],
	 devicedata_.triangles,
	 devicedata_.readyflag,
	 delaunay_.num_triangles);
	cudaCheck(cudaDeviceSynchronize());

	if (!readyflag[0])	{ handleError("Error in triangulation: printing"); }
}

} // namespace delaunay
} // namespace particlesystem
