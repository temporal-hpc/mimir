#include "particlesystem_delaunay.h"

#include <math_constants.h> // CUDART_PI

#include <fstream>
#include <iostream>
#include <map> // std::map
#include <vector> // std::vector

namespace particlesystem::delaunay {

void ParticleSystemDelaunay::handleError(std::string message)
{
	std::cerr << message << std::endl;
	saveFile("error.txt");
	saveMesh2d("error.off");
	exit(EXIT_FAILURE);
}

void ParticleSystemDelaunay::readFile(std::string filename)
{
	std::ifstream input(filename);
	if (input.fail())
	{
		std::cerr << "Could not open file." << std::endl;
		exit(EXIT_FAILURE);
	}

	// Read vertex, face and edge count
	unsigned int vertexCount;
	input >> vertexCount >> delaunay_.num_triangles >> delaunay_.num_edges;
	if (params_.num_elements != vertexCount)
	{
		std::cerr << "The number of vertices on the file does not match the input."
				  << std::endl;
		exit(EXIT_FAILURE);
	}

	delaunay_.triangles = new unsigned int[delaunay_.num_triangles * 3];
	delaunay_.edge_idx = new int2[delaunay_.num_edges];
	delaunay_.edge_ta = new int2[delaunay_.num_edges];
	delaunay_.edge_tb = new int2[delaunay_.num_edges];
	delaunay_.edge_op = new int2[delaunay_.num_edges];

	// Read vertex (particles) data
	for (unsigned int i = 0; i < params_.num_elements; i++)
	{
		input >> positions_[2*i] >> positions_[2*i+1]
		      >> charges_[2*i] >> charges_[2*i+1];
	}

	for (unsigned int i = 0; i < delaunay_.num_triangles; i++)
	{
		input >> delaunay_.triangles[3*i] >> delaunay_.triangles[3*i+1]
		      >> delaunay_.triangles[3*i+2];
	}

	for (unsigned int i = 0; i < delaunay_.num_edges; i++)
	{
		input >> delaunay_.edge_idx[i].x >> delaunay_.edge_idx[i].y
			  >> delaunay_.edge_ta[i].x >> delaunay_.edge_ta[i].y
			  >> delaunay_.edge_tb[i].x >> delaunay_.edge_tb[i].y
			  >> delaunay_.edge_op[i].x >> delaunay_.edge_op[i].y;
	}

	input.close();
}

// DEPRECATED
int ParticleSystemDelaunay::type(double alpha, double mu)
{
    if(alpha == params_.alpha[0] && mu == params_.mu[0]) return 3;
    else if (alpha == params_.alpha[1] && mu == params_.mu[1]) return -1;
    //else if (alpha == params_.alpha[2] && mu == params_.mu[2]) return 1;
    //else if (alpha == params_.alpha[3] && mu == params_.mu[3]) return -3;
    else
    {
        fprintf(stderr, "Particle does not belong to any type: (%lf %lf)\n", alpha, mu);
        exit(EXIT_FAILURE);
    }
}

void ParticleSystemDelaunay::saveConfig()
{
	syncWithDevice();

	std::ofstream output("Config.tmp", std::ios_base::trunc);

	for (unsigned int i = 0; i < params_.num_elements; i++)
	{
		output << positions_[2*i] << " " << positions_[2*i+1] << " "
			   << type(charges_[2*i], charges_[2*i+1]) << " "
			   << velocities_[2*i] << " " << velocities_[2*i+1] << "\n";
	}

	output.close();
	rename("Config.tmp", "Config.fin");
}

void ParticleSystemDelaunay::save_config_counter()
{
	syncWithDevice();

	std::ofstream output("Config.tmp", std::ios_base::trunc);
	for (unsigned int i = 0; i < params_.num_elements; i++)
	{
		output << positions_[2*i] << " "
			   << positions_[2*i+1] << "\n";
	}
	output.close();

	rename("Config.tmp", "Config.fin");
}

void ParticleSystemDelaunay::saveFile(std::string filename)
{
	syncWithDevice();

	std::ofstream output(filename);
	output << params_.num_elements << " "
		   << delaunay_.num_triangles << " "
		   << delaunay_.num_edges << "\n";

	for (unsigned int i = 0; i < params_.num_elements; i++)
	{
		output << positions_[2*i] << " "
			   << positions_[2*i+1] << " "
			   << charges_[2*i] << " "
			   << charges_[2*i+1] << "\n";
	}

	for (unsigned int i = 0; i < delaunay_.num_triangles; i++)
	{
		output << delaunay_.triangles[3*i] << " "
			   << delaunay_.triangles[3*i+1] << " "
			   << delaunay_.triangles[3*i+2] << "\n";;
	}

	for (unsigned int i = 0; i < delaunay_.num_edges; i++)
	{
		output << delaunay_.edge_idx[i].x << " " << delaunay_.edge_idx[i].y << " "
			   << delaunay_.edge_ta[i].x << " " << delaunay_.edge_ta[i].y << " "
			   << delaunay_.edge_tb[i].x << " " << delaunay_.edge_tb[i].y << " "
			   << delaunay_.edge_op[i].x << " " << delaunay_.edge_op[i].y << "\n";
	}

	output.close();
}


void ParticleSystemDelaunay::saveMesh2d(std::string filename)
{
	syncWithDevice();

	std::vector<double2> vertices;
	vertices.reserve(params_.num_elements);
	for (unsigned int i = 0; i < params_.num_elements; i++)
	{
		vertices.push_back(make_double2(positions_[2*i], positions_[2*i+1]));
	}

	std::vector<unsigned int> faces;
	faces.reserve(3 * delaunay_.num_triangles);
	for (unsigned int i = 0; i < delaunay_.num_triangles; i++)
	{
		faces.insert(faces.end(), delaunay_.triangles + 3*i, delaunay_.triangles + 3*i+3);
	}

	std::map< std::pair<int, int>, int > pairs;

	double l = params_.boxlength;
	double lhalf = l / 2.0;

	int j_sec[3] = {0, 0, 1};
	int k_sec[3] = {1, 2, 2};
	int j, k, iv1, iv2;
	double2 pj, pk, s;

	for (unsigned int i = 0; i < delaunay_.num_triangles; i++)
	{
		for (int q = 0; q < 3; q++)
		{
			j = j_sec[q], k = k_sec[q];
			iv1 = faces[3*i + j];
			iv2 = faces[3*i + k];

			std::pair <int, int> jk = std::make_pair(iv1, iv2);
			pj = vertices[iv1];
			pk = vertices[iv2];

			double2 r = make_double2(pk.x - pj.x, pk.y - pj.y);

			if (abs(r.x) > lhalf || abs(r.y) > lhalf)
			{
				if (pairs.find(jk) == pairs.end())
				{
					q = -1;
					pairs[jk] = vertices.size();
					if (r.x > lhalf || r.y > lhalf)
					{
						s = make_double2(pj.x, pj.y);
						if (r.x > lhalf) s.x += l;
						if (r.y > lhalf) s.y += l;
						faces[3*i + j] = pairs[jk];
					}
					else if (r.x < -lhalf || r.y < -lhalf)
					{
						s = make_double2(pk.x, pk.y);
						if (r.x < -lhalf) s.x += l;
						if (r.y < -lhalf) s.y += l;
						faces[3*i + k] = pairs[jk];
					}
					vertices.push_back(s);
				}
				else
				{
					if (r.x > lhalf || r.y > lhalf)
					{
						faces[3*i + j] = pairs[jk];
					}
					if (r.x < -lhalf || r.y < -lhalf)
					{
						faces[3*i + k] = pairs[jk];
					}
				}
			}
		}
	}

	unsigned int num_elements = vertices.size();
	std::ofstream output(filename);

	// Write vertex, face and edge count
	output << "OFF\n"
		   << num_elements << " "
		   << delaunay_.num_triangles << " "
		   << delaunay_.num_edges << "\n";

	// Write vertex (particles) data
	for (unsigned int i = 0; i < num_elements; i++)
	{
		double2 v = vertices[i];
		// Substract (boxLength / 2) for centering the grid at (0, 0)
		output << v.x - lhalf << " " << v.y - lhalf << " " << 0.0 << "\n";
	}

	// Write face (triangles) data
	for (unsigned int i = 0; i < delaunay_.num_triangles; i++)
	{
		output << 3 << " "
			   << faces[3*i] << " "
			   << faces[3*i+1] << " "
			   << faces[3*i+2] << "\n";
	}

	output.close();
}

void ParticleSystemDelaunay::saveMeshSurf(std::string filename)
{
	syncWithDevice();

	std::ofstream output(filename);

	// Write vertex, face and edge count
	output << "OFF\n"
		   << params_.num_elements << " "
		   << delaunay_.num_triangles << " "
		   << delaunay_.num_edges << "\n";

	double xi, yi, zi, phi, theta;
	double r = params_.boxlength / (2 * CUDART_PI);

	// Write vertex (particles) data
	for (unsigned int i = 0; i < params_.num_elements; i++)
	{
		// Map the 2D simulation box onto the 3D flat torus surface
		phi = positions_[2*i] / r;
		theta = positions_[2*i+1] / r;
		xi = r * cos(phi) + r * cos(phi) * cos(theta);
		yi = r * sin(phi) + r * sin(phi) * cos(theta);
		zi = r * sin(theta);
		output << xi << " " << yi << " " << zi << "\n";
	}

	// Write face (triangles) data
	for (unsigned int i = 0; i < delaunay_.num_triangles; i++)
	{
		output << 3 << " "
			   << delaunay_.triangles[3*i] << " "
			   << delaunay_.triangles[3*i+1] << " "
			   << delaunay_.triangles[3*i+2] << "\n";
	}
	output.close();
}

} // namespace particlesystem::delaunay
