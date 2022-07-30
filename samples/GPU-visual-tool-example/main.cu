#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <curand_kernel.h>

#include <chrono> // std::chrono
#include <thread> // std::thread

#include "cudaview/vk_engine.hpp"

#define VEL 0.1
#define BSIZE 256

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)

__global__ void kernel_init(int n, int seed, float2 *points, curandState *state){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < n){
        /* Each thread gets same seed, a different sequence number, no offset */
        curandState localState = state[tid];
        curand_init(seed, tid, 0, &localState);
        /* initialize points with first value */
        points[tid] = make_float2(curand_uniform(&localState), curand_uniform(&localState));
        state[tid] = localState;
    }
}

__global__ void kernel_random_movement(int n, float2 *points, curandState *state) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid == 0) printf("[DEBUG] simulation kernel\n");
    if(tid >= n){
        return;
    }
    curandState localState = state[tid];
    // randoms in between {-0.5, 0.5}
    float dx = VEL*(curand_uniform(&localState) - 0.5);
    float dy = VEL*(curand_uniform(&localState) - 0.5);
    //printf("[thread %i] dx=%f, dy=%f\n", tid, dx, dy);
    float2 p = points[tid];
    if(p.x + dx <= 1.0f && p.x + dx >= -1.0f)
        p.x += dx;
    if(p.y + dy <= 1.0f && p.y + dy >= -1.0f)
        p.y += dy;

    points[tid] = p;
    /* Copy state back to global memory */
    state[tid] = localState;
}

int main(int argc, char *argv[]) {
    // I) INIT (toma de argumentos, creacion de arreglos, etc)
    if(argc != 4){
        fprintf(stderr, "run as ./prog n seed steps\n");
        exit(EXIT_FAILURE);
    }
    int n = atoi(argv[1]);
    int seed = atoi(argv[2]);
    int steps = atoi(argv[3]);
    dim3 b(BSIZE, 1, 1);
    dim3 g((n + b.x - 1)/b.x, 1, 1);

    curandState *dStates;
    float2 *dPoints, *hPoints;

    /* Allocate space for points and prngs */
    hPoints = (float2*)calloc(n, sizeof(float2));

    // habria que llamar un
    //CUDA_CALL(cudaMalloc((void **)&dPoints, n * sizeof(float2)));
    CUDA_CALL(cudaMalloc((void **)&dStates, n * sizeof(curandState)));

    int width = 900, height = 900;
    VulkanEngine engine;

    // [VULKAN] I) CREAR UNA VENTANA VULKAN
    // FLIB_crearVentanaAsync(WIDTH, HEIGHT, ...)
    // [OTRA OPCION] FLIB_crearVentanaSync(WIDTH, HEIGHT, ...)
    // En este momento, la ventana podria aparecer (en negro, sin datos aun)
    engine.init(width, height);

    // [VULKAN] II) "PASAR LOS DATOS AL VISUALIZADOR"
    // FLIB_linkData(&dPoints);
    // [OPCIONAL, SI FUESE 'SYNC'] franciscoLIB_updateWindow(&dPoints);
    // En este momento, la ventana podria verse con el contenido de 'dPoints'
    ViewParams params;
    params.element_count = n;
    params.element_size = sizeof(float2);
    params.data_domain = DataDomain::Domain2D;
    params.resource_type = ResourceType::UnstructuredBuffer;
    params.primitive_type = PrimitiveType::Points;
    engine.addView((void**)&dPoints, params);

    /* SIMULATION */
    kernel_init<<<g, b>>>(n, seed, dPoints, dStates);
    cudaDeviceSynchronize();

    engine.displayAsync();

    for(int i = 0; i < steps; i++) {
        // simulation step (SI FUESE VULKAN-ASYNC, entonces cada modificacion en
        // 'dPoints' se ve refleada inmediatamente en la ventana async)
        std::this_thread::sleep_for(std::chrono::seconds(1));
        engine.prepareWindow();

        kernel_random_movement<<<g, b>>>(n, dPoints, dStates);
        cudaDeviceSynchronize();
        // [OPCIONAL, SI FUESE 'SYNC'] franciscoLIB_updateWindow(&dPoints);
        engine.updateWindow();

        #ifdef DEBUG
            printf("[DEBUG] simulation step %i:\n", i);
            CUDA_CALL(cudaMemcpy(hPoints, dPoints, n * sizeof(float2), cudaMemcpyDeviceToHost));
            for(int j=0; j<n; ++j){
                printf("[DEBUG] P[%i] = {%f, %f}\n", j, hPoints[j].x, hPoints[j].y);
            }
            getchar();
        #endif
    }

    /* Copy device memory to host and show result */
    CUDA_CALL(cudaMemcpy(hPoints, dPoints, n * sizeof(float2), cudaMemcpyDeviceToHost));
    for(int i=0; i<n; ++i){
        printf("P[i] = {%f, %f}\n", hPoints[i].x, hPoints[i].y);
    }

    /* Cleanup */
    CUDA_CALL(cudaFree(dStates));
    CUDA_CALL(cudaFree(dPoints));
    free(hPoints);

    return EXIT_SUCCESS;
}
