/*
 * particlesystemdelaunay.h
 *
 *  Created on: 15-04-2016
 *      Author: francisco
 */

#ifndef PARTICLESYSTEMDELAUNAY_H_
#define PARTICLESYSTEMDELAUNAY_H_

#include <vector_types.h>

#include <memory> // std::unique_ptr
#include <string> // std::string

#include "particlesystem.h"
#include "simparameters.h"
#include "../src/device_data.h"

#include <mimir/mimir.hpp>

// TODO: Encapsulate triangulation data into a struct

namespace particlesystem {
namespace delaunay {

struct Triangulation
{
	unsigned int num_triangles;
	unsigned int num_edges;  // Number of faces and edges in the triangulation respectively
	unsigned int* triangles; // Array of triangles in (p1, p2, p3) index format
	int2* edge_idx;          // Array of pairs of indices that make edges in the triangulation
	int2* edge_ta;
	int2* edge_tb;           // Indices of triangles Ta and Tb for each edge respectively
	int2* edge_op;           // Indices of the particles opposite to each edge in the triangulation
};

class ParticleSystemDelaunay: ParticleSystem
{
public:
	typedef std::unique_ptr<ParticleSystemDelaunay> Ptr;

	ParticleSystemDelaunay(SimParameters p);

	/* Constructor that initializes the random number generator structures with
	 * a specific seed value.
	 */
	ParticleSystemDelaunay(SimParameters p, unsigned long long seed);

	/* Constructor for loading particle and delaunay mesh data from a previously
	 * saved file.
	 */
	ParticleSystemDelaunay(SimParameters p, unsigned long long seed,
	                       std::string filename);

	/* Public destructor; actual cleanup is done by the private finalize()
	 * function.
	 */
	~ParticleSystemDelaunay();

	/* Executes [numIterations] time steps, transferring data from device to
	 * host once for each [save_period] time steps.
	 */
	void runSimulation(int num_iter, unsigned int save_period);

	/* Dumps the content of all device arrays to a plain text file, whose name
	 * can be passed as an argument to the Particle System constructor.
	 */
	void saveFile(std::string filename);

	/* Saves the positions of each particle to an .off file, which contains the
	 * 2D representation of the underlying 2D periodical Delaunay triangulation.
	 */
	void saveMesh2d(std::string filename);

	/* Saves the positions of each particle to an .off file, which contains the
	 * flat torus representation of the underlying periodical Delaunay
	 * triangulation.
	 */
	void saveMeshSurf(std::string filename);

	void readFile(std::string filename);

	/* Saves the current configuration of the system, containing the position
	 * and type of each particle. The output file can be used to obtain a simple
	 * 2D visualization of the particle system.
	 */
	void saveConfig();
	void save_config_counter();

private:
	/* Executes a single time step simulation.
	 */
	void runTimestep();

	/* Initialization routines common to all constructors.
	 */
	void initCommon();

	/* Initializes the kernel parameters.
	 */
	void initCuda(unsigned long long seed);

	/* Initializes the positions of the particle system on host memory.
	 */
	void initTriangulationHost();

	/* Transfers from host to device memory the data structures necessary
	 * for parallel execution of the particle system simulation.
	 */
	void loadOnDevice();

	/* Transfers data from host to device memory.
	 */
	void syncWithDevice();

	/* Handles a simulation error, dumping the current particle data and
	 * printing the error message before clean-up of the data structures.
	 */
	void handleError(std::string message);

	/* Returns the integer representing the type of the particle with the
	 * inputted (alpha, mu) value pair.
	 */
	int type(double alpha, double mu);

	void integrateShared();    // Integration step with shared memory for n-body simulation
	void integrateShuffle();   // Integration step using the __shfl() instruction for n-body simulation.
	void updateDelaunay();     // Delaunay edge flip algorithm
	void correctOverlaps();    // Overlap correction
	void updateTriangles();    // Inverted triangle correction
	void checkTriangulation(); // [Optional] Check for triangulation validity on device memory

	SimParameters params_;
	Triangulation delaunay_;
	double* velocities_;
	unsigned int current_read;  // Index of the device particles array that is being used for read operations
	unsigned int current_write; // Index of the device particles array that is being used for write operations
	int* readyflag;             // Host variable mapped to device memory for communication with kernels

	uint2 hParticleLaunch; // Kernel launch data for thread <-> particle mappings
	uint2 hTriangleLaunch; // Kernel launch data for thread <-> triangle mappings
	uint2 hEdgeLaunch;     // Kernel launch data for thread <-> edge mappings

	unsigned int overlap_counter; // Counter for total number of overlap correction iterations in the simulation
	unsigned int flip_counter;    // Counter for total number of edge flip iterations in the simulation

	// should be *hDeviceData if using multiple gpus
	DeviceData devicedata_;
    mimir::InstanceHandle engine;
    mimir::AllocHandle interop[6];
    mimir::ViewHandle particle_views[2];
    mimir::ViewHandle edge_views[2];
};

} // namespace delaunay
} // namespace particlesystem

#endif /* PARTICLESYSTEMDELAUNAY_H_ */
