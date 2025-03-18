/*
 * particlesystemdelaunay.h
 *
 *  Created on: 15-04-2016
 *      Author: francisco
 */

#ifndef PARTICLESYSTEMDELAUNAY_H_
#define PARTICLESYSTEMDELAUNAY_H_

#include <vector_types.h>

#include <memory>
#include <string>

#include "../src/devicedata_verlet.h"
#include "base/particlesystem.h"
#include "base/simparameters.h"

namespace particlesystem {
namespace verlet {

class ParticleSystemVerlet: ParticleSystem
{
public:
	typedef std::unique_ptr<ParticleSystemVerlet> Ptr;

	ParticleSystemVerlet(SimParameters p, VerletParameters vp);

	/* Constructor that initializes the random number generator structures with
	 * a specific seed value.
	 */
	ParticleSystemVerlet(SimParameters p, VerletParameters vp,
	                     unsigned long long seed);

	/* Constructor for loading particle and delaunay mesh data from a previously
	 * saved file.
	 */
	ParticleSystemVerlet(SimParameters p, VerletParameters vp,
	                     unsigned long long seed, std::string filename);

	/* Public destructor; actual cleanup is done by the private finalize()
	 * function.
	 */
	~ParticleSystemVerlet();

	/* Executes [numIterations] time steps, transferring data from device to
	 * host once for each [infoPeriod] time steps.
	 */
	void runSimulation(int num_iter, unsigned int save_period);

	/* Dumps the content of all device arrays to a plain text file, whose name
	 * can be passed as an argument to the Particle System constructor.
	 */
	void saveFile(std::string filename);

	void readFile(std::string filename);

	/* Saves the current configuration of the system, containing the position
	 * and type of each particle. The output file can be used to obtain a simple
	 * 2D visualization of the particle system.
	 */
	void save_config();
	void save_config_counter();

private:
	/* Executes a single time step simulation.
	 */
	void runTimestep();

	/* Initialization routines common to all constructors.
	 */
	void initCommon();

	/* Initializes the kernel parameters
	 */
	void initCuda(unsigned long long seed);

	/* Transfers from host to device memory the data structures necessary for
	 * parallel execution of the particle system simulation.
	 */
	void loadOnDevice();

	/* Transfers data from host to device memory
	 */
	void syncWithDevice();

	/* Tears down the data structures in host and device memory, cleaning the
	 * device state after all the device memory has been freed.
	 */
	void finalize();

	/* Handles a simulation error, dumping the current particle data and
	 * printing the error message before clean-up of the data structures.
	 */
	void handle_error(std::string message);

	/* Returns the integer representing the type of the particle with the
	 * inputted (alpha, mu) value pair.
	 */
	int type(double alpha, double mu);

	void integrate(); // Integration step with shared memory for n-body simulation
	void correctOverlaps(); // Overlap correction
	void buildVerletList(); // Verlet list construction on device memory
	void verifyVerletList(); // Checks if Verlet list needs to be reconstructed

	SimParameters params_;
	VerletParameters verlet_;
	unsigned int current_read;    // Index of the device particles array that is being used for read operations
	unsigned int current_write;   // Index of the device particles array that is being used for write operations
	unsigned int overlap_counter; // Counter for total number of overlap correction iterations in the simulation
	uint2 hParticleLaunch;        // Kernel launch data for thread <-> particle mappings

	int* readyflag; // Host variable mapped to device memory for communication with kernels
	int* hParticleCounter;

	// should be *hDeviceData if using multiple gpus
	DeviceData devicedata_;
};

} // namespace verlet
} // namespace particlesystem

#endif /* PARTICLESYSTEMDELAUNAY_H_ */
