#include "particlesystem/particlesystem_verlet.h"

#include <fstream>
#include <iostream>
#include <map>
#include <vector>

namespace particlesystem {
namespace verlet {

void ParticleSystemVerlet::handle_error(std::string message)
{
	std::cerr << message << std::endl;
	saveFile("error.txt");
	finalize();
	exit(EXIT_FAILURE);
}

void ParticleSystemVerlet::readFile(std::string filename)
{
	std::ifstream input(filename);
	if (input.fail())
	{
		std::cerr << "Could not open file." << std::endl;
		exit(EXIT_FAILURE);
	}

	// Read vertex, face and edge count
	unsigned int vertexCount;
	input >> vertexCount;
	if (params_.num_elements != vertexCount)
	{
		std::cerr << "The number of vertices on the file does not match the input."
				  << std::endl;
		exit(EXIT_FAILURE);
	}

	// Read vertex (particles) data
	for (unsigned int i = 0; i < params_.num_elements; i++)
	{
		input >> positions_[2*i] >> positions_[2*i+1]
		      >> charges_[2*i] >> charges_[2*i+1];
	}

	input.close();
}

int ParticleSystemVerlet::type(double alpha, double mu)
{
    if(alpha == params_.alpha[0] && mu == params_.mu[0]) return 3;
    else if (alpha == params_.alpha[1] && mu == params_.mu[1]) return -1;
    else if (alpha == params_.alpha[2] && mu == params_.mu[2]) return 1;
    else if (alpha == params_.alpha[3] && mu == params_.mu[3]) return -3;
    else
    {
        fprintf(stderr, "Particle does not belong to any type: (%lf %lf)\n", alpha, mu);
        return 0;
    }
}

void ParticleSystemVerlet::save_config()
{
	syncWithDevice();

	std::ofstream output("Config.tmp", std::ios_base::trunc);

	for (unsigned int i = 0; i < params_.num_elements; i++)
	{
		output << positions_[2*i] << " "
			   << positions_[2*i+1] << " "
			   << type(charges_[2*i], charges_[2*i+1]) << "\n";
	}

	output.close();

	rename("Config.tmp", "Config.fin");
}

void ParticleSystemVerlet::save_config_counter()
{
	syncWithDevice();

	std::ofstream output("Config.tmp", std::ios_base::trunc);

	for (unsigned int i = 0; i < params_.num_elements; i++)
	{
		output << positions_[2*i] << " "
			   << positions_[2*i+1] << " "
			   << hParticleCounter[i] << "\n";
	}

	output.close();

	rename("Config.tmp", "Config.fin");
}

void ParticleSystemVerlet::saveFile(std::string filename)
{
	syncWithDevice();

	std::ofstream output(filename);
	output << params_.num_elements << "\n";

	for (unsigned int i = 0; i < params_.num_elements; i++)
	{
		output << positions_[2*i] << " "
			   << positions_[2*i+1] << " "
			   << charges_[2*i] << " "
			   << charges_[2*i+1] << "\n";
	}

	output.close();
}

} // namespace verlet
} // namespace particlesystem
