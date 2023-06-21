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

#include <cudaview/vk_engine.hpp>

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

    int width = 900, height = 900;
    VulkanEngine engine;
    engine.init(width, height);

    // CREACION DE DATOS
    printf("Inicializando.................."); fflush(stdout);
    t1 = omp_get_wtime();
    init_prob(n, original, seed, prob);

    /*engine.createViewStructured((void**)&d1, extent, sizeof(int),
      DataDomain::Domain3D, DataFormat::Int32, StructuredDataType::Voxels
    );*/
    /*engine.createViewStructured((void**)&d2, extent, sizeof(int),
      DataDomain::Domain3D, DataFormat::Int32, StructuredDataType::Voxels
    );*/

    ViewParams params;
    // TODO CAMBIAR A 2D
    params.element_count = n*n*n;
    params.element_size = sizeof(int);

    // TODO: 2D --> {n, n, 1}
    params.extent = {(unsigned)n, (unsigned)n, (unsigned)n};
    // TODO: CAMBIAR A DOMAIN 2D
    params.data_domain = DataDomain::Domain3D;
    params.resource_type = ResourceType::StructuredBuffer;
    params.primitive_type = PrimitiveType::Voxels;

    //params.resource_type = ResourceType::TextureLinear;
    //params.texture_format = TextureFormat::Int32;
    auto v1 = engine.createView((void**)&d1, params);
    /*engine.createViewStructured((void**)&d2, extent, sizeof(int),
      DataDomain::Domain3D, DataFormat::Int32, StructuredDataType::Texture
    );*/
    //gpuErrchk(cudaMalloc(&d1, sizeof(int)*n*n*n));
    // TODO CAMBIAR A 2D
    //gpuErrchk(cudaMalloc(&d2, sizeof(int)*n*n*n));
    params.options.visible = false;
    auto v2 = engine.createView((void**)&d2, params);
    //std::cout << v1 << " " << v2 << "\n";
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
            engine.prepareWindow();

            kernel_CA3D<<<grid, block>>>(n, d1, d2);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );

            v1->toggleVisibility();
            v2->toggleVisibility();

            engine.updateWindow();

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
