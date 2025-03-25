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
#include "validation.hpp" // checkCuda
using namespace mimir;

int main(int argc, char **argv){
    if(argc != 8){
        fprintf(stderr, "ejecutar como ./prog n nt B seed steps prob modo\nmodo = 0 CPU,  1 GPU\nB <= 10 (blocksize is BxBxB)\n\n");
        exit(EXIT_FAILURE);
    }
    const char *map[2] = {"CPU", "GPU"};
    // args
    long n      = atoi(argv[1]);
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
    EngineHandle engine = nullptr;
    createEngine(width, height, &engine);

    AllocHandle ping = nullptr, pong = nullptr, colormap = nullptr;
    allocLinear(engine, (void**)&d1, sizeof(int) * n*n*n, &ping);
    allocLinear(engine, (void**)&d2, sizeof(int) * n*n*n, &pong);

    float4 *d_colors = nullptr;
    float4 h_colors[2] = { {1,1,1,0.5}, {0,0,1,1} };
    unsigned int num_colors = std::size(h_colors);
    auto color_bytes = sizeof(float4) * num_colors;
    allocLinear(engine, (void**)&d_colors, color_bytes, &colormap);
    gpuErrchk(cudaMemcpy(d_colors, h_colors, color_bytes, cudaMemcpyHostToDevice));

    auto grid_layout = Layout::make(n, n, n);
    uint32_t index_count = n * n * n;
    ViewHandle v1 = nullptr, v2 = nullptr;
    ViewDescription desc{
        .view_type   = ViewType::Voxels,
        .domain_type = DomainType::Domain3D,
        .attributes  = {
            { AttributeType::Position, makeStructuredGrid(engine, grid_layout) },
            { AttributeType::Color, {
                .source   = colormap,
                .size     = num_colors,
                .format   = FormatDescription::make<float4>(),
                .indexing = {
                    .source     = ping,
                    .size       = index_count,
                    .index_size = sizeof(int),
                }
            }}
        },
        .layout       = grid_layout,
        .default_size = 10.f,
    };
    createView(engine, &desc, &v1);

    desc.attributes[AttributeType::Color].indexing.source = pong;
    desc.visible = false;
    createView(engine, &desc, &v2);

    // TODO CAMBIAR A 2D
    gpuErrchk(cudaMemcpy(d1, original, sizeof(int)*n*n*n, cudaMemcpyHostToDevice));
    printf("done: %f secs\n", omp_get_wtime() - t1);

    displayAsync(engine);

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
            prepareViews(engine);

            kernel_CA3D<<<grid, block>>>(n, d1, d2);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );

            toggleVisibility(v1);
            toggleVisibility(v2);

            updateViews(engine);

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
    destroyEngine(engine);
}
