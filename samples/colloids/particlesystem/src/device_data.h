/*
 * simparameters.h
 *
 *  Created on: 06-04-2016
 *      Author: francisco
 */

#ifndef DEVICE_DATA_H_
#define DEVICE_DATA_H_

/* Structure for holding the simulation parameters, which hold constant during
 * the lifetime of each simulation instance. The structure is sent to device
 * constant memory when the simulation instance is created.
 */

#include <curand_kernel.h>

#define BLOCK_SIZE_PARTICLES 	128
#define BLOCK_SIZE_TRIANGLES	256
#define BLOCK_SIZE_EDGES		512

namespace particlesystem {
namespace delaunay {

struct DeviceData
{
	double2* positions[2];   // Buffer for particle positions
	double2* charges;       // Array of (alpha, mu) pairs for each particle
	double2* velocities;
    int* types;              // Particle types corresponding to (alpha,mu) tuples
    float4* colors;
	curandState* rng_states; // Array of random number generator states
	unsigned int* triangles; // Array of particle indices that conform each triangle
	int2* edge_idx;          // Array of pairs of indices that make edges in the triangulation
	int2* edge_ta;
	int2* edge_tb;           // Indices of triangles Ta and Tb for each edge respectively
	int2* edge_op;           // Indices of the particles opposite to each edge in the triangulation
	int* dTriRel;
	int* dTriReserv;         // Auxiliary arrays for the edge flip algorithm
	int* readyflag;          // Flag for communication between host program flow and device code
};

} // namespace delaunay
} // namespace particlesystem

#endif /* DEVICE_DATA_H_ */
