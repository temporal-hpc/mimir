/*
 * globalvars.h
 *
 *  Created on: 24-03-2016
 *      Author: francisco
 */

#ifndef CONFIGVARS_H_
#define CONFIGVARS_H_

// Number of iterations
#define NUM_ITER 10000

// Diameter of each particle
#define SIGMA 1.0

// Value of the simulation time step
#define DT 0.01

// Raduis of the skin for neighbor list checking
#define SKIN			3.0

// Cutoff radius for two-body force calculation
#define CUTOFF			2.5

// Maximum mumber of neighbors per particle
#define MAX_NEIGHBORS	40

// Maximum number of particles per cell
#define CELL_SIZE		20

// RNG seed value (for debugging)
#define RNG_SEED 2763872ULL

#endif /* CONFIGVARS_H_ */
