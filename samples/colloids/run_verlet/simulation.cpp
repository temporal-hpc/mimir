#include "particlesystem/particlesystem_verlet.h"
#include "configvars.h"

#include <iostream>
#include <sstream>

int main(int argc, char **argv)
{
	using namespace particlesystem::verlet;

	int numArgs = 4 + 3 * (NUM_TYPES);
	if (argc < numArgs)	// Get values
	{
		std::cerr << "./modelo N L T c_a alpha_a mu_a c_b alpha_b mu_b c_c "
				  << "alpha_c mu_c c_d alpha_d mu_d\n"
				  << "USAGE:\n"
				  << "N: number of simulated particles\n"
				  << "L: length of the simulation square box\n"
				  << "c_i: Concentration ratio of particle i, between 0 and 1\n"
				  << "alpha_i, mu_i: interaction parameters of particle i\n"
				  << std::endl;
		exit(EXIT_FAILURE);
	}

	particlesystem::SimParameters p;

	std::stringstream input; // Get input parameters
	for (int i = 1; i < argc; i++) { input << argv[i] << " "; }

	// Put the parameters on the SimParameters structure
	input >> p.num_elements >> p.boxlength >> p.diffusion;
	for (int i = 0; i < NUM_TYPES; i++)
	{
		input >> p.conc[i] >> p.alpha[i] >> p.mu[i];
	}

	p.timestep = DT;
	p.radius = SIGMA;

	particlesystem::verlet::VerletParameters vp;
	double skin = SKIN;
	double cutoff = CUTOFF;
	vp.cell_size = CELL_SIZE;
	vp.max_neighbors = MAX_NEIGHBORS;
	vp.cutoff_sqr = cutoff * cutoff;
	vp.skin_sqr = skin * skin;
	vp.deltaskin_sqr = 0.25 * (skin - cutoff) * (skin - cutoff);
	vp.grid_size = (unsigned int) (p.boxlength / skin);
	vp.cell_length = p.boxlength / vp.grid_size;

	// Create an instance of a particle system simulation
	ParticleSystemVerlet::Ptr ps(new ParticleSystemVerlet(p, vp));
	// Run the simulation
	ps->runSimulation(NUM_ITER / 5, 1);
	// Save final results to output files
	ps->saveFile("out.txt");

	std::cout << "Simulation ended successfully." << std::endl;
	return(EXIT_SUCCESS);
}
