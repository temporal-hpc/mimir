/*
 * globalvars.h
 *
 *  Created on: 24-03-2016
 *      Author: francisco
 */

#ifndef CONFIGVARS_H_
#define CONFIGVARS_H_

#define NUM_ITER 10000

#define SIGMA 1.0				// Diameter of each particle
#define DT 0.01					// Value of the simulation time step

#define SKIN			3.0		// Raduis of the skin for neighbor list checking
#define CUTOFF			2.5		// Cutoff radius for two-body force calculation
#define MAX_NEIGHBORS	40		// Maximum mumber of neighbors per particle
#define CELL_SIZE		20		// Maximum number of particles per cell

#define RNG_SEED 2763872ULL		// RNG seed value (for debugging)

#endif /* CONFIGVARS_H_ */
