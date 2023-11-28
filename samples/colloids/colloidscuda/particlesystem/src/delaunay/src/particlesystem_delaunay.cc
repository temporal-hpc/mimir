#include "particlesystem/particlesystem_delaunay.h"

#include <math_constants.h> // CUDART_PI

#include <cstdio>

#include "base/cuda_check.h"

namespace particlesystem {
namespace delaunay {

using namespace mimir;

ParticleSystemDelaunay::ParticleSystemDelaunay(SimParameters p)
	: ParticleSystemDelaunay(p, time(nullptr))
{}

ParticleSystemDelaunay::ParticleSystemDelaunay(SimParameters p,
                                               unsigned long long seed):
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
                                               unsigned long long seed,
                                               std::string pFileName):
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

void ParticleSystemDelaunay::runSimulation(int num_iter,
                                           unsigned int save_period=0)
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
    engine.prepareViews();

	integrateShuffle();
	updateTriangles();
	checkTriangulation();
	updateDelaunay();
	correctOverlaps();

    views[current_read]->toggleVisibility();
    views[current_write]->toggleVisibility();

    engine.updateViews();
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

float4 getTypeColor(int type)
{
    switch (type)
    {
        case 0: return {27.f / 255,158.f / 255,119.f / 255,1};
        case 1: return {217.f / 255,95.f / 255,2.f / 255,1};
        case 2: return {117.f / 255,112.f / 255,179.f / 255,1};
        case 3: return {231.f / 255,41.f / 255,138.f / 255,1};
        default: return {0, 0, 0, 1};
    }
}

void ParticleSystemDelaunay::loadOnDevice()
{
    std::vector<float4> colors;
    colors.reserve(params_.num_elements);
    for (unsigned int i = 0; i < params_.num_elements; ++i)
    {
        colors.push_back(getTypeColor(types_[i]));
    }

    ViewerOptions viewer_opts;
    viewer_opts.window_title = "Colloid"; // Top-level window.
    viewer_opts.window_size = {1920, 1080};
    viewer_opts.present = PresentOptions::Immediate;
    engine.init(viewer_opts);

	// Load particle data
	size_t pos_bytes = params_.num_elements * 2 * sizeof(double);
	//cudaCheck(cudaMalloc(&devicedata_.positions[0], pos_bytes));
	//cudaCheck(cudaMalloc(&devicedata_.positions[1], pos_bytes));
    unsigned l = params_.boxlength;

    MemoryParams m;
    m.layout          = DataLayout::Layout1D;
    m.element_count.x = params_.num_elements;
    m.data_type       = DataType::Double;
    m.channel_count   = 2;
    m.resource_type   = ResourceType::Buffer;
    interop[current_read] = engine.createBuffer((void**)&devicedata_.positions[current_read], m);
    interop[current_write] = engine.createBuffer((void**)&devicedata_.positions[current_write], m);

    m.data_type       = DataType::Int;
    m.channel_count   = 1;
    interop[2] = engine.createBuffer((void**)&devicedata_.types, m);
    //m.data_type       = DataType::Float;
    //m.channel_count   = 4;
    //interop[2] = engine.createBuffer((void**)&devicedata_.colors, m);

    ViewParams2 v;
    v.element_count = params_.num_elements;
    v.extent        = {l, l, 1};
    v.data_domain   = DataDomain::Domain2D;
    v.domain_type   = DomainType::Unstructured;
    v.view_type     = ViewType::Markers;
    v.attributes[AttributeType::Position] = *interop[current_read];
    v.attributes[AttributeType::Color] = *interop[2];
    v.options.default_size = 100.f;
    /*v.options.external_shaders = {
        {"shaders/marker_vertexMain.spv", VK_SHADER_STAGE_VERTEX_BIT},
        {"shaders/marker_geometryMain.spv", VK_SHADER_STAGE_GEOMETRY_BIT},
        {"shaders/marker_fragmentMain.spv", VK_SHADER_STAGE_FRAGMENT_BIT}
    };*/
    views[current_read] = engine.createView(v);

    v.options.visible = false;
    v.attributes[AttributeType::Position] = *interop[current_write];
    views[current_write] = engine.createView(v);

	cudaCheck(cudaMemcpy(devicedata_.positions[current_read], positions_,
			             pos_bytes, cudaMemcpyHostToDevice));

	cudaCheck(cudaMalloc(&devicedata_.velocities, pos_bytes));
	cudaCheck(cudaMemcpy(devicedata_.velocities, velocities_,
			             pos_bytes, cudaMemcpyHostToDevice));

	// Load particle type data
    auto type_bytes = sizeof(int) * params_.num_elements;
    //cudaCheck(cudaMalloc(&devicedata_.types, type_bytes));
    cudaCheck(cudaMemcpy(devicedata_.types, types_, type_bytes, cudaMemcpyHostToDevice));

    auto color_bytes = sizeof(float4) * params_.num_elements;
    cudaCheck(cudaMalloc(&devicedata_.colors, color_bytes));
    cudaCheck(cudaMemcpy(devicedata_.colors, colors.data(), color_bytes, cudaMemcpyHostToDevice));

	cudaCheck(cudaMalloc(&devicedata_.charges, pos_bytes));
	cudaCheck(cudaMemcpy(devicedata_.charges, charges_, pos_bytes,
			             cudaMemcpyHostToDevice));

	// Load edge data
	size_t edge_bytes = delaunay_.num_edges * 2 * sizeof(int);

	cudaCheck(cudaMalloc(&devicedata_.edge_idx, edge_bytes));
	cudaCheck(cudaMemcpy(devicedata_.edge_idx, delaunay_.edge_idx, edge_bytes,
			             cudaMemcpyHostToDevice));

	cudaCheck(cudaMalloc(&devicedata_.edge_ta, edge_bytes));
	cudaCheck(cudaMemcpy(devicedata_.edge_ta, delaunay_.edge_ta, edge_bytes,
			             cudaMemcpyHostToDevice));

	cudaCheck(cudaMalloc(&devicedata_.edge_tb, edge_bytes));
	cudaCheck(cudaMemcpy(devicedata_.edge_tb, delaunay_.edge_tb, edge_bytes,
			             cudaMemcpyHostToDevice));

	cudaCheck(cudaMalloc(&devicedata_.edge_op, edge_bytes));
	cudaCheck(cudaMemcpy(devicedata_.edge_op, delaunay_.edge_op, edge_bytes,
			             cudaMemcpyHostToDevice));

	// Load triangle data
	size_t triangle_bytes = delaunay_.num_triangles * 3 * sizeof(unsigned int);

	cudaCheck(cudaMalloc(&devicedata_.triangles, triangle_bytes));
	cudaCheck(cudaMemcpy(devicedata_.triangles, delaunay_.triangles,
			             triangle_bytes, cudaMemcpyHostToDevice));

	// Allocate auxiliary triangle arrays
	size_t aux_triangle_bytes = delaunay_.num_triangles * sizeof(int);

	cudaCheck(cudaMalloc(&devicedata_.dTriRel, aux_triangle_bytes));
	cudaCheck(cudaMalloc(&devicedata_.dTriReserv, aux_triangle_bytes));

	// Zero-copy auxiliary variable for edge-flip algorithm
	cudaCheck(cudaHostAlloc(&readyflag, sizeof(int), cudaHostAllocMapped));
	cudaCheck(cudaHostGetDevicePointer(&devicedata_.readyflag, readyflag, 0));

	printf("Triangulation loaded to device memory.\n");
    engine.displayAsync();
}

void ParticleSystemDelaunay::syncWithDevice()
{
	size_t pos_bytes = params_.num_elements * 2 * sizeof(double);
	cudaCheck(cudaMemcpy(positions_, devicedata_.positions[current_read],
			             pos_bytes, cudaMemcpyDeviceToHost));
	cudaCheck(cudaMemcpy(velocities_, devicedata_.velocities,
			             pos_bytes, cudaMemcpyDeviceToHost));
	// No need to synchronize particle type data, since it stays the same
	// throughout the simulation.

	size_t triangle_bytes = delaunay_.num_triangles * 3 * sizeof(unsigned int);
	cudaCheck(cudaMemcpy(delaunay_.triangles, devicedata_.triangles,
			             triangle_bytes, cudaMemcpyDeviceToHost));

	size_t edge_bytes = delaunay_.num_edges * 2 * sizeof(int);
	cudaCheck(cudaMemcpy(delaunay_.edge_idx, devicedata_.edge_idx, edge_bytes,
			             cudaMemcpyDeviceToHost));
	cudaCheck(cudaMemcpy(delaunay_.edge_ta, devicedata_.edge_ta, edge_bytes,
			             cudaMemcpyDeviceToHost));
	cudaCheck(cudaMemcpy(delaunay_.edge_tb, devicedata_.edge_tb, edge_bytes,
			             cudaMemcpyDeviceToHost));
	cudaCheck(cudaMemcpy(delaunay_.edge_op, devicedata_.edge_op, edge_bytes,
			             cudaMemcpyDeviceToHost));
}

} // namespace delaunay
} // namespace particlesystem
