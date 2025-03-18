/*
 * simparameters.h
 *
 *  Created on: 06-04-2016
 *      Author: francisco
 */

#ifndef DEVICE_DATA_H_
#define DEVICE_DATA_H_

/* Structure for holding the simulation parameters, which hold constant during the lifetime of each simulation instance.
 * The structure is sent to device constant memory when the simulation instance is created.
 */

#include <curand_kernel.h>

#define BLOCK_SIZE_PARTICLES 	128
#define BLOCK_SIZE_TRIANGLES	256
#define BLOCK_SIZE_EDGES		512

namespace particlesystem {
namespace verlet {

typedef struct
{
	/* Parameters used in the verlet list procedures */
	unsigned int grid_size;     // Number of cells in the grid
	unsigned int cell_size;     // Maximum number of particles per cell
	unsigned int max_neighbors; // Maximum number of neighbors per particle
	double cutoff_sqr;          // Square of cutoff on short range forces calculation
	double cell_length;         // Length of each cell box on the simulation grid
	double skin_sqr;            // Square of the "cell skin" value
	double deltaskin_sqr;       // Square of the difference between skin and particle diameter
}
VerletParameters;


struct DeviceData
{
	double2* positions[3];   // Buffer for particle positions
	double2* charges;       // Array of (alpha, mu) pairs for each particle
	curandState* rng_states; // Array of random number generator states

	int* neighbor_counters;  // Array of neighbor list sizes for each particle
	int* neighbor_list;      // List of neighbors for each particle
	int* cell_counters;      // Array of cell list sizes for each particle
	int* cell_list;          // List of particles held in each cell

	int* readyflag;          // Flag for communication between host program flow and device code
};

} // namespace verlet
} // namespace particlesystem

#endif /* DEVICE_DATA_H_ */
