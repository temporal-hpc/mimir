
#ifndef RANDOM_NUMBER_GENERATOR_H
#define RANDOM_NUMBER_GENERATOR_H

/* Class for a pseudo-random number generator with uniform distribution, using
 * the cuRAND host API. The generator is created with a fixed amount of random
 * numbers generated on the GPU. After all numbers are used, that generator
 * instance cannot be used anymore and must be deleted.
 */

// TODO: Documentation

namespace particlesystem {

template <typename T> class RandGen
{
public:
	RandGen(unsigned int num_elements, unsigned long long seed);
	~RandGen();
	T next();

private:
	unsigned int index_, maxidx_;
	T *data_;
};

} // namespace particlesystem

#endif
