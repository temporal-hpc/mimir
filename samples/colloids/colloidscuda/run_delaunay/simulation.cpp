#include "particlesystem/particlesystem_delaunay.h"
#include "configvars.h"

#include <iostream>
#include <sstream>

int main(int argc, char **argv)
{
	using namespace particlesystem::delaunay;

	int numArgs = 4 + 3 * (NUM_TYPES);
	if (argc < numArgs)	// Get values
	{
		std::cerr << "./modelo N L T c_a alpha_a mu_a c_b alpha_b mu_b ...\n"
				  << "USAGE:\n"
				  << "N: number of simulated particles\n"
				  << "L: length of the simulation square box\n"
				  << "c_i: Concentration ratio of particle i, between 0 and 1\n"
				  << "alpha_i, mu_i: charges of particle i\n"
				  << std::endl;
		exit(EXIT_FAILURE);
	}

	particlesystem::SimParameters p;
	p.timestep = DT;
	p.radius = SIGMA;

	std::stringstream input; // Get input parameters
	for (int i = 1; i < argc; i++) { input << argv[i] << " "; }
	// Put the parameters on the SimParameters structure
	input >> p.num_elements >> p.boxlength >> p.diffusion;
	for (int i = 0; i < NUM_TYPES; i++)
	{
		input >> p.conc[i] >> p.alpha[i] >> p.mu[i];
	}
	initParameters(p);

	// Create an instance of a particle system simulation
	ParticleSystemDelaunay::Ptr ps(new ParticleSystemDelaunay(p));
	// Run the simulation.
	ps->runSimulation(NUM_ITER, 2000);
	// Save final results to output file.
	ps->saveFile("out.txt");

	std::cout << "Simulation ended successfully." << std::endl;
	return(EXIT_SUCCESS);
}
