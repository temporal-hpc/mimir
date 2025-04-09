/*
 * simparameters.h
 *
 *  Created on: Jan 20, 2018
 *      Author: francisco
 */

#ifndef SIMPARAMETERS_H_
#define SIMPARAMETERS_H_

#define NUM_TYPES 2 // Number of particle types

namespace particlesystem {

typedef struct
{
	unsigned int num_elements; // Number of simulated particles.
	double boxlength; // Length of simulation box.
	double diffusion; // Diffusion coefficient.
	double radius; // Diameter of each particle.
	double timestep; // Value of each simulation time-step.
	double radius_sqr; // Square of diameter, precomputed for use on device code.
	double sqrt_DTD; // Square root of (timestep * diffusion), precomputed for use on device code.
	double conc[NUM_TYPES]; // Concentrations of each particle type.
	double alpha[NUM_TYPES]; // Alpha charge for each particle type.
	double mu[NUM_TYPES]; // Mu charge for each particle type.
}
SimParameters;

void initParameters(SimParameters &params);
void initParticles(double* positions, double* alpha_mu, int *types,
    SimParameters params, unsigned long long seed
);

} // namespace particlesystem

#endif /* SIMPARAMETERS_H_ */
