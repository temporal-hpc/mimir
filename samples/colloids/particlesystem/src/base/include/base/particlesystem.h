/*
 * particlesystem.h
 *
 *  Created on: Jan 18, 2018
 *      Author: francisco
 */

#ifndef PARTICLESYSTEM_H_
#define PARTICLESYSTEM_H_

// TODO: Add hdf5 support for faster and lighter I/O

namespace particlesystem {

class ParticleSystem
{
public:
	virtual ~ParticleSystem() {};
	virtual void runSimulation(int num_iter, unsigned int save_period) = 0;
protected:
	virtual void runTimestep() = 0;

	double* positions_; // Array of particle descriptions (x, y, alpha, mu)
	double* charges_;
    int* types_;
};

} // namespace particlesystem

#endif /* PARTICLESYSTEM_H_ */
