#include "particlesystem_delaunay.h"

#include <math_constants.h> // CUDART_PI

#include <cstdio>
#include <iostream> // std::cin

#include "cuda_check.h"

namespace particlesystem {
namespace delaunay {

using namespace mimir;

ParticleSystemDelaunay::ParticleSystemDelaunay(SimParameters p)
	: ParticleSystemDelaunay(p, time(nullptr))
{}

ParticleSystemDelaunay::ParticleSystemDelaunay(SimParameters p, unsigned long long seed):
	params_(p),
	current_read(0),
	current_write(1),
	readyflag(nullptr),
	overlap_counter(0),
	flip_counter(0)
{
	positions_ = new double[params_.num_elements * 2];
	charges_ = new double[params_.num_elements * 2];
	velocities_ = new double[params_.num_elements * 2];
    types_ = new int[params_.num_elements];

	initParticles(positions_, charges_, types_, params_, seed);
	initTriangulationHost();
	initCommon();
	initCuda(seed);
	loadOnDevice();
}

ParticleSystemDelaunay::ParticleSystemDelaunay(SimParameters p,
    unsigned long long seed, std::string pFileName):
	params_(p),
	current_read(0),
	current_write(1),
	readyflag(nullptr),
	overlap_counter(0),
	flip_counter(0)
{
	positions_ = new double[params_.num_elements * 2];
	charges_ = new double[params_.num_elements * 2];
	velocities_ = new double[params_.num_elements * 2];
    types_ = new int[params_.num_elements];

	readFile(pFileName);
	initCommon();
	initCuda(seed);
	loadOnDevice();
}

ParticleSystemDelaunay::~ParticleSystemDelaunay()
{
	cudaCheck(cudaFree(devicedata_.positions[0]));
	cudaCheck(cudaFree(devicedata_.positions[1]));
	cudaCheck(cudaFree(devicedata_.rng_states));
	cudaCheck(cudaFree(devicedata_.triangles));
	cudaCheck(cudaFree(devicedata_.edge_idx));
	cudaCheck(cudaFree(devicedata_.edge_ta));
	cudaCheck(cudaFree(devicedata_.edge_tb));
	cudaCheck(cudaFree(devicedata_.edge_op));
	cudaCheck(cudaFree(devicedata_.dTriRel));
	cudaCheck(cudaFree(devicedata_.dTriReserv));
	cudaCheck(cudaFreeHost(readyflag));

	destroyInstance(instance);
	cudaCheck(cudaDeviceReset());

	delete [] positions_;
	delete [] charges_;
	delete [] velocities_;
    delete [] types_;
	delete [] delaunay_.triangles;
	delete [] delaunay_.edge_idx;
	delete [] delaunay_.edge_ta;
	delete [] delaunay_.edge_tb;
	delete [] delaunay_.edge_op;
}

void ParticleSystemDelaunay::runSimulation(int num_iter, unsigned int save_period=0)
{
	int iter;
	float time = 0.0f;

	cudaEvent_t start, stop;
	cudaCheck(cudaEventCreate(&start, cudaEventDefault));
	cudaCheck(cudaEventCreate(&stop, cudaEventDefault));
	cudaCheck(cudaEventRecord(start, 0));

	for (iter = 1; iter <= num_iter; iter++)
	{
		// Print the current iteration, and sync data with host
		if (save_period != 0 && iter % save_period == 0)
		{
			printf("Iter = %d; t = %f\n", iter, iter * params_.timestep);
			saveConfig();
		}

		runTimestep();
	}

	cudaCheck(cudaEventRecord(stop, 0));
	cudaCheck(cudaEventSynchronize(stop));
	cudaCheck(cudaEventElapsedTime(&time, start, stop));
	cudaCheck(cudaEventDestroy(start));
	cudaCheck(cudaEventDestroy(stop));

	printf("Average simulation time (ms): %f\n"
		   "Total simulation time (ms): %f\n"
		   "Average overlap correction iterations: %f\n"
		   "Average edge flip iterations: %f\n",
		   time / iter, time, static_cast<float>(overlap_counter) / time,
		   static_cast<float>(flip_counter) / iter);
}

void ParticleSystemDelaunay::runTimestep()
{
    prepareViews(instance);

	integrateShuffle();
	updateTriangles();
	checkTriangulation();
	updateDelaunay();
	correctOverlaps();

    toggleVisibility(particle_views[current_read]);
    toggleVisibility(particle_views[current_write]);

    toggleVisibility(edge_views[current_read]);
    toggleVisibility(edge_views[current_write]);

    updateViews(instance);
}

void ParticleSystemDelaunay::initCommon()
{
	double particle_area = pow(params_.radius / 2, 2) * CUDART_PI;
	double box_area = params_.boxlength * params_.boxlength;
	double packing = (params_.num_elements * particle_area) / box_area;

	printf("Simulation parameters:\n"
		   "Particle count: %u\n"
		   "Packing fraction: %lf\n"
		   "Box length: %lf\n"
		   "Edge count: %u\n"
		   "Triangle count: %u\n",
		   params_.num_elements, packing, params_.boxlength,
		   delaunay_.num_edges, delaunay_.num_triangles);

	unsigned int blockSizeP = BLOCK_SIZE_PARTICLES;
	unsigned int blockSizeT = BLOCK_SIZE_TRIANGLES;
	unsigned int blockSizeE = BLOCK_SIZE_EDGES;

	hParticleLaunch = make_uint2( (params_.num_elements + blockSizeP - 1) / blockSizeP, blockSizeP);
	hTriangleLaunch = make_uint2( (delaunay_.num_triangles + blockSizeT - 1) / blockSizeT, blockSizeT );
	hEdgeLaunch = make_uint2( (delaunay_.num_edges + blockSizeE - 1) / blockSizeE, blockSizeE );

	printf("Setting kernel parameters (grid size, block size):\n"
		   "Particles = (%u, %u)\n"
		   "Triangles = (%u, %u)\n"
		   "Edges = (%u, %u)\n",
	       hParticleLaunch.x, hParticleLaunch.y,
	       hTriangleLaunch.x, hTriangleLaunch.y,
	       hEdgeLaunch.x, hEdgeLaunch.y);
}

void ParticleSystemDelaunay::loadOnDevice()
{
    ViewerOptions viewer_opts;
    viewer_opts.window.title = "Colloids"; // Top-level window.
    viewer_opts.window.size  = {1920, 1080};
    viewer_opts.present.mode = PresentMode::Immediate;
	viewer_opts.background_color = {1.f, 1.f, 1.f, 1.f};
    createInstance(viewer_opts, &instance);

	// Load particle data
	size_t pos_bytes = params_.num_elements * sizeof(double2);

    // Particle positions
    allocLinear(instance, (void**)&devicedata_.positions[current_read], pos_bytes, &interop[current_read]);
    allocLinear(instance, (void**)&devicedata_.positions[current_write], pos_bytes, &interop[current_write]);

    // Velocities
    allocLinear(instance, (void**)&devicedata_.velocities, pos_bytes, &interop[2]);

    // Particle types / colors
    allocLinear(instance, (void**)&devicedata_.types, sizeof(int) * params_.num_elements, &interop[3]);
    allocLinear(instance, (void**)&devicedata_.colors, sizeof(float4) * NUM_TYPES, &interop[5]);

    ViewDescription vp;
    vp.layout = Layout::make(params_.num_elements);
    vp.domain = DomainType::Domain2D;
    vp.type   = ViewType::Markers;
	vp.scale  = {0.1f, 0.1f, 0.1f};
	vp.position  = {0.f, 0.f, 0.f};
	vp.linewidth = 0.f;
    vp.attributes[AttributeType::Position] =
	{
		.source  = interop[current_read],
		.size    = params_.num_elements,
		.format  = FormatDescription::make<double2>(),
	};
    vp.attributes[AttributeType::Color] =
	{
		.source   = interop[5],
		.size     = NUM_TYPES,
		.format   = FormatDescription::make<float4>(),
		.indexing = {
			.source     = interop[3],
			.size       = params_.num_elements,
			.index_size = sizeof(int),
		}
	};
    createView(instance, &vp, &particle_views[current_read]);

    vp.visible = false;
    vp.attributes[AttributeType::Position].source = interop[current_write];
    createView(instance, &vp, &particle_views[current_write]);

    // Edges
    allocLinear(instance, (void**)&devicedata_.triangles, sizeof(int3) * delaunay_.num_triangles, &interop[4]);

	MeshOptions options{ .periodic = true, };

    ViewDescription vpe;
    vpe.layout = Layout::make(params_.boxlength, params_.boxlength);
	vpe.options = options;
    vpe.domain = DomainType::Domain2D;
    vpe.type   = ViewType::Edges;
	vpe.scale  = {0.1f, 0.1f, 0.1f};
	vpe.position  = {0.f, 0.f, 0.001f};
	vpe.linewidth = 0.f;
    vpe.attributes[AttributeType::Position] =
	{
		.source   = interop[current_read],
		.size     = params_.num_elements,
		.format   = FormatDescription::make<double2>(),
		.indexing = {
			.source     = interop[4],
			.size       = delaunay_.num_triangles * 3,
			.index_size = sizeof(uint),
		}
	};
    createView(instance, &vpe, &edge_views[current_read]);

	vpe.visible = false;
    vpe.attributes[AttributeType::Position].source = interop[current_write];
    createView(instance, &vpe, &edge_views[current_write]);

	// Original allocations; replaced by Mimir interop functions
	//cudaCheck(cudaMalloc(&devicedata_.positions[0], pos_bytes));
	//cudaCheck(cudaMalloc(&devicedata_.positions[1], pos_bytes));
	//cudaCheck(cudaMalloc(&devicedata_.velocities, pos_bytes));
    //cudaCheck(cudaMalloc(&devicedata_.types, type_bytes));
    //cudaCheck(cudaMalloc(&devicedata_.colors, color_bytes));
	//cudaCheck(cudaMalloc(&devicedata_.triangles, triangle_bytes));

	cudaCheck(cudaMemcpy(devicedata_.positions[current_read], positions_, pos_bytes, cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(devicedata_.velocities, velocities_, pos_bytes, cudaMemcpyHostToDevice));

	// Load particle type data
    auto type_bytes = sizeof(int) * params_.num_elements;
    cudaCheck(cudaMemcpy(devicedata_.types, types_, type_bytes, cudaMemcpyHostToDevice));

    std::vector<float4> h_colors{
        { 49, 130, 189, 1 },
        { 49, 163, 84, 1 },
        { 158, 202, 225, 1 },
	};
	for (auto& c : h_colors) { c.x /= 255.f; c.y /= 255.f; c.z /= 255.f; }
    auto color_bytes = sizeof(float4) * NUM_TYPES;
    cudaCheck(cudaMemcpy(devicedata_.colors, h_colors.data(), color_bytes, cudaMemcpyHostToDevice));

	cudaCheck(cudaMalloc(&devicedata_.charges, pos_bytes));
	cudaCheck(cudaMemcpy(devicedata_.charges, charges_, pos_bytes, cudaMemcpyHostToDevice));

	// Load edge data
	size_t edge_bytes = delaunay_.num_edges * 2 * sizeof(int);

	cudaCheck(cudaMalloc(&devicedata_.edge_idx, edge_bytes));
	cudaCheck(cudaMemcpy(devicedata_.edge_idx, delaunay_.edge_idx, edge_bytes, cudaMemcpyHostToDevice));

	cudaCheck(cudaMalloc(&devicedata_.edge_ta, edge_bytes));
	cudaCheck(cudaMemcpy(devicedata_.edge_ta, delaunay_.edge_ta, edge_bytes, cudaMemcpyHostToDevice));

	cudaCheck(cudaMalloc(&devicedata_.edge_tb, edge_bytes));
	cudaCheck(cudaMemcpy(devicedata_.edge_tb, delaunay_.edge_tb, edge_bytes, cudaMemcpyHostToDevice));

	cudaCheck(cudaMalloc(&devicedata_.edge_op, edge_bytes));
	cudaCheck(cudaMemcpy(devicedata_.edge_op, delaunay_.edge_op, edge_bytes, cudaMemcpyHostToDevice));

	// Load triangle data
	size_t triangle_bytes = delaunay_.num_triangles * 3 * sizeof(unsigned int);
	cudaCheck(cudaMemcpy(devicedata_.triangles, delaunay_.triangles, triangle_bytes, cudaMemcpyHostToDevice));

	// Allocate auxiliary triangle arrays
	size_t aux_triangle_bytes = delaunay_.num_triangles * sizeof(int);

	cudaCheck(cudaMalloc(&devicedata_.dTriRel, aux_triangle_bytes));
	cudaCheck(cudaMalloc(&devicedata_.dTriReserv, aux_triangle_bytes));

	// Zero-copy auxiliary variable for edge-flip algorithm
	cudaCheck(cudaHostAlloc(&readyflag, sizeof(int), cudaHostAllocMapped));
	cudaCheck(cudaHostGetDevicePointer(&devicedata_.readyflag, readyflag, 0));

	printf("Triangulation loaded to device memory.\n");
	setCameraPosition(instance, {0.f, 0.f, -15.f});
    displayAsync(instance);

	printf("Press ENTER to start the simulation...\n");
    std::cin.get(); // For recording
}

void ParticleSystemDelaunay::syncWithDevice()
{
	size_t pos_bytes = params_.num_elements * 2 * sizeof(double);
	cudaCheck(cudaMemcpy(positions_, devicedata_.positions[current_read], pos_bytes, cudaMemcpyDeviceToHost));
	cudaCheck(cudaMemcpy(velocities_, devicedata_.velocities, pos_bytes, cudaMemcpyDeviceToHost));
	// No need to synchronize particle type data, since it stays the same throughout the simulation.

	size_t triangle_bytes = delaunay_.num_triangles * 3 * sizeof(unsigned int);
	cudaCheck(cudaMemcpy(delaunay_.triangles, devicedata_.triangles, triangle_bytes, cudaMemcpyDeviceToHost));

	size_t edge_bytes = delaunay_.num_edges * 2 * sizeof(int);
	cudaCheck(cudaMemcpy(delaunay_.edge_idx, devicedata_.edge_idx, edge_bytes, cudaMemcpyDeviceToHost));
	cudaCheck(cudaMemcpy(delaunay_.edge_ta, devicedata_.edge_ta, edge_bytes, cudaMemcpyDeviceToHost));
	cudaCheck(cudaMemcpy(delaunay_.edge_tb, devicedata_.edge_tb, edge_bytes, cudaMemcpyDeviceToHost));
	cudaCheck(cudaMemcpy(delaunay_.edge_op, devicedata_.edge_op, edge_bytes, cudaMemcpyDeviceToHost));
}

} // namespace delaunay
} // namespace particlesystem
