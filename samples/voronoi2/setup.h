#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <curand.h>
#include <math.h>
#include <chrono>
#include <random>
#include <vector>
#include <omp.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include "utils.h"
#include "nbody.cuh"

#define EPSILON 0.001
#define METHOD "dJFA"

const double G = 6.6743e-11;
#ifdef DLSS
#include "../DLSS/include/nvsdk_ngx.h"
#include "../DLSS/include/nvsdk_ngx_defs.h"
#include "../DLSS/include/nvsdk_ngx_params.h"
#include "../DLSS/include/nvsdk_ngx_helpers.h"
#endif

#define ll long long

#include <mimir/mimir.hpp>
using namespace mimir;

struct Setup{
    int N;
    int S;
    int mode;
    int iters;
    int device;
    int mem_gpu;
    int block_size;
    int distance_function;
    int redux;
    int seed;
    int mu;
    int sample;
    int molecules;
    int comparison;
    int pbc;
    int k;
    int k_m;
    int k_r;
    int k_rm;
    float L_avg;
    float rL_avg;

    int *v_diagram;
    int *seeds;
    int *areas;

    int N_p;
    int *redux_v_diagram;
    int *backup_v_diagram;

    int *gpu_v_diagram;
    int *gpu_seeds;
    int *gpu_redux_seeds;
    int *gpu_delta;
    int *gpu_delta_max;
    int *gpu_areas;

    int *gpu_redux_vd;
    int *gpu_backup_vd;

    dim3 normal_block;
    dim3 normal_grid;
    dim3 seeds_block;
    dim3 seeds_grid;
    dim3 redux_block;
    dim3 redux_grid;
    curandState *r_device;

    //Nbody params
    //double G = 6.6743e-11;  // gravitational constant
    double DT = 1.0;  // time step
    double M = 1.0; // particle mass
    double *seeds_vel;
    double *gpu_seeds_vel;

    float3 *gpu_seed_colors = nullptr;
    float4 *gpu_vd_colors = nullptr;
    CudaviewEngine *engine;
};

void initialize_variables(Setup *setup, int N, int S, int mode, int iters, int device, int mem_gpu, int block_size, int distance_function, int redux,int mu, int sample, int molecules, int comparison){
    setup->N = N;
    setup->S = S;
    setup->mode = mode;
    setup->iters = iters;
    setup->device = device;
    setup->mem_gpu = mem_gpu;
    setup->block_size = block_size;
    setup->distance_function = distance_function;
    setup->redux = redux;
    setup->seed = 0;
    setup->sample = sample;
    setup->molecules = molecules;
    setup->comparison = comparison;
    setup->pbc = (sample>=1)? 1 : 0;

    setup->mu = mu;
    setup->N_p = N/mu;
    setup->L_avg = sqrt(setup->N * setup->N / setup->S);
    setup->rL_avg = sqrt(setup->N_p * setup->N_p / setup->S);

    setup->k = pow(2,int(log2(setup->N)));
    setup->k_m = pow(2,int(log2(2 * setup->L_avg)) + 1);
    setup->k_r = pow(2,int(log2(setup->N_p)));
    setup->k_rm = pow(2, int(log2(2 * setup->rL_avg)) + 1);

    setup->normal_block = dim3 (setup->block_size,setup->block_size,1);
    setup->normal_grid = dim3 ((setup->N + setup->block_size + 1)/setup->block_size, (setup->N + setup->block_size + 1)/setup->block_size,1);
    setup->seeds_block = dim3 (setup->block_size,1,1);
    setup->seeds_grid = dim3 ((setup->S + setup->block_size + 1)/setup->block_size,1,1);
    setup->redux_block = dim3 (setup->block_size,setup->block_size,1);
    setup->redux_grid = dim3 ((setup->N_p + setup->block_size + 1)/setup->block_size, (setup->N_p + setup->block_size + 1)/setup->block_size,1);
}

void allocate_arrays(Setup *setup){
    setup->engine = new CudaviewEngine();
    setup->engine->init(1920, 1080);

    MemoryParams m1;
    m1.layout         = DataLayout::Layout2D;
    m1.element_count  = {(unsigned)setup->N, (unsigned)setup->N, 1};
    m1.component_type = ComponentType::Int;
    m1.channel_count  = 1;
    m1.resource_type  = ResourceType::Buffer;
    auto grid = setup->engine->createBuffer((void**)&setup->gpu_backup_vd, m1);

    m1.component_type = ComponentType::Float;
    m1.channel_count  = 4;
    auto grid_colors = setup->engine->createBuffer((void**)&setup->gpu_vd_colors, m1);

    ViewParams p1;
    p1.element_count = setup->N * setup->N;
    p1.extent        = {(uint)setup->N, (uint)setup->N, 1};
    p1.data_domain   = DataDomain::Domain2D;
    p1.domain_type   = DomainType::Structured;
    p1.view_type     = ViewType::Voxels;
    p1.attributes[AttributeType::Color] = *grid_colors;
    p1.options.default_size = 1.f;
    p1.options.custom_val = setup->S;
    /*p1.options.external_shaders = {
        {"shaders/voxel_vertexImplicitMain.spv", VK_SHADER_STAGE_VERTEX_BIT},
        {"shaders/voxel_geometryMain2D.spv", VK_SHADER_STAGE_GEOMETRY_BIT},
        {"shaders/voxel_fragmentMain.spv", VK_SHADER_STAGE_FRAGMENT_BIT}
    };*/
    setup->engine->createView(p1);

    MemoryParams m2;
    m2.layout         = DataLayout::Layout1D;
    m2.element_count  = {(unsigned)setup->S, 1, 1};
    m2.component_type = ComponentType::Int;
    m2.channel_count  = 1;
    m2.resource_type  = ResourceType::Buffer;
    auto seeds = setup->engine->createBuffer((void**)&setup->gpu_seeds, m2);

    ViewParams p2;
    p2.element_count = setup->S;
    p2.extent        = {(uint)setup->N, (uint)setup->N, 1};
    p2.data_domain   = DataDomain::Domain2D;
    p2.domain_type   = DomainType::Unstructured;
    p2.view_type     = ViewType::Markers;
    p2.attributes[AttributeType::Position] = *seeds;
    p2.options.default_size = setup->N / 100.f;
    p2.options.default_color = {0,0,1,1};
    /*p2.options.external_shaders = {
        {"shaders/marker_vertexMain.spv", VK_SHADER_STAGE_VERTEX_BIT},
        {"shaders/marker_geometryMain.spv", VK_SHADER_STAGE_GEOMETRY_BIT},
        {"shaders/marker_fragmentMain.spv", VK_SHADER_STAGE_FRAGMENT_BIT}
    };*/
    setup->engine->createView(p2);

    setup->v_diagram = (int*)malloc(setup->N*setup->N*sizeof(int));
    setup->backup_v_diagram = (int*)malloc(setup->N*setup->N*sizeof(int));
    setup->seeds = (int*)malloc(setup->S*sizeof(int));
    setup->areas = (int*)malloc(setup->S*sizeof(int));
    setup->redux_v_diagram = (int*)malloc(setup->N_p*setup->N_p*sizeof(int));
    setup->seeds_vel = (double*)malloc(2*setup->S*sizeof(double));

    cudaMalloc((void**)&setup->r_device, setup->S * sizeof(curandState));
    cudaMalloc(&setup->gpu_v_diagram, setup->N*setup->N*sizeof(int));
    //cudaMalloc(&setup->gpu_backup_vd, setup->N*setup->N*sizeof(int));
    cudaMalloc(&setup->gpu_redux_vd,setup->N_p*setup->N_p*sizeof(int));
    //cudaMalloc(&setup->gpu_seeds, setup->S*sizeof(int));
    cudaMalloc(&setup->gpu_delta, setup->S*sizeof(int));
    cudaMalloc(&setup->gpu_seeds_vel, setup->S*2*sizeof(double));

    // Allocs used by Mimir (temporary)
    cudaMalloc(&setup->gpu_seed_colors, setup->S*sizeof(float3));
}

void setDeviceInfo(Setup *setup){
    if(setup->sample==2){
        for (int it=0; it<setup->S; it++) {
            setup->seeds_vel[it*2] = 0.0;//double(rand()%10000)/10000.0;
            setup->seeds_vel[it*2+1] = 0.0;//double(rand()%10000)/2500.0;
        }
    }

    cudaMemcpy(setup->gpu_seeds_vel, setup->seeds_vel, 2*setup->S*sizeof(double), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    cudaMemcpy(setup->gpu_seeds, setup->seeds, setup->S*sizeof(int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
}

void printRunInfo(Setup *setup){
    printf("MU: %i\n", setup->mu);
    printf("Sample: %i\n", setup->sample);
    printf("Comparison: %i\n", setup->comparison);
    printf("PBC: %i\n", setup->pbc);
    printf("Distance used(1: Manhattan, 0:Euclidean): %i\n", setup->distance_function);
    printf("k: %i, k_m: %i, k_r: %i, k_rm: %i\n", setup->k, setup->k_m, setup->k_r, setup->k_rm);
    if(setup->sample == 2){
        printf("G: %.12f\n", G);
        printf("DT: %f\n", setup->DT);
        printf("M: %f\n", setup->M);
    }
}

void seetSeeds(Setup *setup){
    if(setup->sample == 0)initSeeds(setup->seeds, setup->N, setup->S);
    else read_coords(setup->seeds, setup->N, setup->S, 0, setup->molecules);
}

void getDeviceArrays(Setup *setup){
    cudaMemcpy(setup->v_diagram, setup->gpu_v_diagram, setup->N*setup->N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaMemcpy(setup->backup_v_diagram, setup->gpu_backup_vd, setup->N*setup->N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaMemcpy(setup->redux_v_diagram, setup->gpu_redux_vd, setup->N_p*setup->N_p*sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
}

void freeSpace(Setup *setup){
    cudaFree(setup->gpu_v_diagram);
    cudaFree(setup->gpu_backup_vd);
    cudaFree(setup->gpu_redux_vd);
    cudaFree(setup->gpu_seeds);
    cudaFree(setup->gpu_seeds_vel);
    cudaFree(setup->r_device);
    cudaFree(setup->gpu_seed_colors);

    free(setup->v_diagram);
    free(setup->backup_v_diagram);
    free(setup->redux_v_diagram);
    free(setup->seeds);
    free(setup->seeds_vel);
}

void writeGridColors(Setup *setup);