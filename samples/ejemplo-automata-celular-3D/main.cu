#include <cuda.h>
#include <cstdlib>
#include <cstdio>
#include <omp.h>
#define CA_LOW 2
#define CA_HIGH 3
#define CA_NACER 3

#include "tools.h"
#include "kernel3D.cuh"
#include "openmp3D.h"

#include "cudaview/vk_engine.hpp"

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
    printf("modo: %s     n=%i (%.3f GiBytes / cubo)    nt=%i   B=%i  steps=%i\n", map[modo], n, sizeof(int)*n*n*n/(1024*1024*1024.0), nt, B, steps);

    // original (3D)
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

    uint3 extent{(unsigned)n, (unsigned)n, (unsigned)n};
    engine.addViewStructured((void**)&d1, extent, sizeof(int),
      DataDomain::Domain3D, DataFormat::Int32
    );
    /*engine.addViewStructured((void**)&d2, extent, sizeof(int),
      DataDomain::Domain3D, DataFormat::Int32
    );*/
    //gpuErrchk(cudaMalloc(&d1, sizeof(int)*n*n*n));
    gpuErrchk(cudaMalloc(&d2, sizeof(int)*n*n*n));
    gpuErrchk(cudaMemcpy(d1, original, sizeof(int)*n*n*n, cudaMemcpyHostToDevice));
    printf("done: %f secs\n", omp_get_wtime() - t1);

    engine.displayAsync();
    engine.updateWindow();

    // ejecucion
    print_cube(n, original, "INPUT");
    printf("Press Enter...\n"); fflush(stdout); getchar();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // OJO: la cantidad total de threads tiene que ser Bx * By * Bz <= 1024
    dim3 block(B,B,B);
    dim3 grid((n+block.x-1)/block.x, (n+block.y-1)/block.y, (n+block.z-1)/block.z);
    if(modo==1){
        // modo GPU
        for(int i=0; i<steps; ++i){
            printf("[GPU] Simulacion step=%i........", i);
            cudaEventRecord(start);

            // llamada al kernel
            std::this_thread::sleep_for(std::chrono::seconds(1));
            engine.prepareWindow();

            kernel_CA3D<<<grid, block>>>(n, d1, d2);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );

            engine.updateWindow();

            // tiempo y print
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&timems, start, stop);
            printf("done: %f\n", timems/1000.0);
            gpuErrchk(cudaMemcpy(original, d2, sizeof(int)*n*n*n, cudaMemcpyDeviceToHost));
            print_cube(n, original, "[GPU] Automata celular");
            printf("Press Enter...\n"); fflush(stdout); getchar();
            std::swap(d1, d2);
        }
    }
    else{
        // secundario CPU (3D)
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
}
