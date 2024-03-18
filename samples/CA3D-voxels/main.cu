#include <cuda.h>
#include <cstdlib>
#include <cstdio>
#include <omp.h>
#define CA_LOW 2
#define CA_HIGH 3
#define CA_NACER 3

#include <iostream>

#include "tools.h"
#include "kernel3D.cuh"
#include "openmp3D.h"

#include <mimir/mimir.hpp>
#include <mimir/validation.hpp> // checkCuda
using namespace mimir;
using namespace mimir::validation; // checkCuda

int main(int argc, char **argv){
    if(argc != 8){
        fprintf(stderr, "ejecutar como ./prog n nt B seed steps prob modo\nmodo = 0 CPU,  1 GPU\nB <= 10 (blocksize is BxBxB)\n\n");
        exit(EXIT_FAILURE);
    }
    const char *map[2] = {"CPU", "GPU"};
    // args
    long n       = atoi(argv[1]);
    int nt      = atoi(argv[2]);
    int B       = atoi(argv[3]);
    int seed    = atoi(argv[4]);
    int steps   = atoi(argv[5]);
    float prob  = atof(argv[6]);
    int modo = atoi(argv[7]);
    float timems;
    double t1;

    if(B > 10 || modo > 1){
        fprintf(stderr, "ejecutar como ./prog n nt B seed steps prob modo\nmodo = 0 CPU,  1 GPU\nB <= 10 (blocksize is BxBxB)\n\n");
        exit(EXIT_FAILURE);
    }

    // SETEO DE OPENMP THREADS (solo relevante para inicializar datos y solucion CPU)

    omp_set_num_threads(nt);
    // TODO CAMBIAR A 2D
    printf("modo: %s     n=%ld (%.3f GiBytes / cubo)    nt=%i   B=%i  steps=%i\n", map[modo], n, sizeof(int)*n*n*n/(1024*1024*1024.0), nt, B, steps);

    // original (3D)
    // TODO CAMBIAR A 2D
    int *original = new int[n*n*n];

    // punteros GPU (3D)
    int *d1, *d2;

    // CREACION DE DATOS
    printf("Inicializando.................."); fflush(stdout);
    t1 = omp_get_wtime();
    init_prob(n, original, seed, prob);

    int width = 1920, height = 1080;
    MimirEngine engine;
    engine.init(width, height);

    MemoryParams mp;
    mp.layout         = DataLayout::Layout3D;
    mp.element_count  = {(uint)n, (uint)n, (uint)n};
    mp.component_type = ComponentType::Int;
    mp.channel_count  = 1;
    mp.resource_type  = ResourceType::Buffer;
    //mp.resource_type  = ResourceType::LinearTexture;
    auto m1 = engine.createBuffer((void**)&d1, mp);
    auto m2 = engine.createBuffer((void**)&d2, mp);

    ViewParams params;
    params.element_count = n * n * n;
    params.extent        = {(unsigned)n, (unsigned)n, (unsigned)n};
    params.data_domain   = DataDomain::Domain3D;
    params.domain_type   = DomainType::Structured;
    params.view_type     = ViewType::Voxels;
    //params.view_type     = ViewType::Image;
    params.attributes[AttributeType::Color] = *m1;
    params.options.default_size = 5.f;
    /*params.options.external_shaders = {
        {"shaders/voxel_vertexImplicitMain.spv", VK_SHADER_STAGE_VERTEX_BIT},
        {"shaders/voxel_geometryMain.spv", VK_SHADER_STAGE_GEOMETRY_BIT},
        {"shaders/voxel_fragmentMain.spv", VK_SHADER_STAGE_FRAGMENT_BIT}
    };*/
    /*params.options.external_shaders = {
        {"shaders/texture_vertex3dMain.spv", VK_SHADER_STAGE_VERTEX_BIT},
        {"shaders/texture_frag3d_Float1.spv", VK_SHADER_STAGE_FRAGMENT_BIT}
    };*/
    auto v1 = engine.createView(params);

    params.attributes[AttributeType::Color] = *m2;
    params.options.visible = false;
    auto v2 = engine.createView(params);

    // TODO CAMBIAR A 2D
    gpuErrchk(cudaMemcpy(d1, original, sizeof(int)*n*n*n, cudaMemcpyHostToDevice));
    printf("done: %f secs\n", omp_get_wtime() - t1);

    engine.displayAsync();

    // ejecucion
    print_cube(n, original, "INPUT");
    printf("Press Enter...\n"); fflush(stdout); getchar();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // OJO: la cantidad total de threads tiene que ser Bx * By * Bz <= 1024
    // TODO CAMBIAR A 2D
    dim3 block(B,B,B);
    // TODO CAMBIAR A 2D
    dim3 grid((n+block.x-1)/block.x, (n+block.y-1)/block.y, (n+block.z-1)/block.z);
    if(modo==1){
        // modo GPU
        for(int i=0; i<steps; ++i){
            printf("[GPU] Simulacion step=%i........", i);
            cudaEventRecord(start);

            // llamada al kernel
            engine.prepareViews();

            kernel_CA3D<<<grid, block>>>(n, d1, d2);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );

            v1->toggleVisibility();
            v2->toggleVisibility();

            engine.updateViews();

            // tiempo y print
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&timems, start, stop);
            printf("done: %f\n", timems/1000.0);
            // TODO CAMBIAR A 2D
            gpuErrchk(cudaMemcpy(original, d2, sizeof(int)*n*n*n, cudaMemcpyDeviceToHost));
            print_cube(n, original, "[GPU] Automata celular");
            printf("Press Enter...\n"); fflush(stdout); getchar();
            std::swap(d1, d2);
        }
    }
    else{
        // secundario CPU (3D)
        // TODO CAMBIAR A 2D
        int *CPUd2 = new int[n*n*n];

        // modo CPU (multicore segun nt escogido)
        for(int i=0; i<steps; ++i){
            printf("[CPU] Simulacion step=%i........", i);
            t1 = omp_get_wtime();

            // llamada a paso de simulacion
            openmp_CA3D(n, original, CPUd2);

            // tiempo y print
            printf("done: %f\n", omp_get_wtime() - t1);
            print_cube(n, CPUd2, "Automata celular (CPU):");
            printf("Press Enter...\n"); fflush(stdout); getchar();
            std::swap(original, CPUd2);
        }
    }
    printf("Finished running all steps\n");
}
