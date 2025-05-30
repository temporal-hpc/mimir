#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <curand_kernel.h>
#include <chrono> // std::chrono
#include <thread> // std::thread

#include <mimir/mimir.hpp>
#include "validation.hpp" // checkCuda
using namespace mimir;

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

    // [VULKAN] I) CREAR UNA VENTANA VULKAN
    // FLIB_crearVentanaAsync(WIDTH, HEIGHT, ...)
    // [OTRA OPCION] FLIB_crearVentanaSync(WIDTH, HEIGHT, ...)
    // En este momento, la ventana podria aparecer (en negro, sin datos aun)
    InstanceHandle instance = nullptr;
    createInstance(900, 900, &instance);

    // [VULKAN] II) "PASAR LOS DATOS AL VISUALIZADOR"
    // FLIB_linkData(&dPoints);
    // [OPCIONAL, SI FUESE 'SYNC'] franciscoLIB_updateViews(&dPoints);
    // En este momento, la ventana podria verse con el contenido de 'dPoints'
    AllocHandle points;
    allocLinear(instance, (void**)&dPoints, sizeof(float2) * n, &points);

    ViewHandle view = nullptr;
    ViewDescription desc
    {
        .type   = ViewType::Markers,
        .domain = DomainType::Domain2D,
        .attributes = {
            { AttributeType::Position, {
                .source = points,
                .size   = static_cast<unsigned int>(n),
                .format = FormatDescription::make<float2>(),
            }}
        },
        .layout       = Layout::make(n),
        .default_size = .01f,
        .linewidth    = .01f,
        .position     = {-0.5, -0.5, 0.f}
    };
    createView(instance, &desc, &view);

    /* SIMULATION */
    kernel_init<<<g, b>>>(n, seed, dPoints, dStates);
    cudaDeviceSynchronize();

    displayAsync(instance);

    for(int i = 0; i < steps; i++) {
        // simulation step (SI FUESE VULKAN-ASYNC, entonces cada modificacion en
        // 'dPoints' se ve refleada inmediatamente en la ventana async)
        std::this_thread::sleep_for(std::chrono::seconds(1));
        prepareViews(instance);

        kernel_random_movement<<<g, b>>>(n, dPoints, dStates);
        cudaDeviceSynchronize();
        // [OPCIONAL, SI FUESE 'SYNC'] franciscoLIB_updateViews(&dPoints);
        updateViews(instance);

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
    destroyInstance(instance);
    free(hPoints);

    return EXIT_SUCCESS;
}
