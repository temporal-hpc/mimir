#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include <chrono>
#include <thread>


// incluir Mimir para la visualizacion de la simulacion en GPU
#include <mimir/mimir.hpp>
#include <mimir/validation.hpp> // checkCuda
using namespace mimir;
using namespace mimir::validation; // checkCuda
using namespace std;

#define BLOCK_SIZE 16  // Tamaño del bloque CUDA
#define NX 512
#define NY 256

__global__ void build_up_b_kernel(float *u, float *v, float *b, float dx, float dy, float dt, float rho) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i < NX - 1 && j > 0 && j < NY - 1) {
        b[j * NX + i] = rho * (1.0f / dt) * (
            (u[j * NX + i + 1] - u[j * NX + i - 1]) / (2.0f * dx) + 
            (v[(j + 1) * NX + i] - v[(j - 1) * NX + i]) / (2.0f * dy)
        ) - 
        ((u[j * NX + i + 1] - u[j * NX + i - 1]) / (2.0f * dx)) * 
        ((u[j * NX + i + 1] - u[j * NX + i - 1]) / (2.0f * dx)) - 
        2.0f * ((u[(j + 1) * NX + i] - u[(j - 1) * NX + i]) / (2.0f * dy)) * 
        ((v[j * NX + i + 1] - v[j * NX + i - 1]) / (2.0f * dx)) -
        ((v[(j + 1) * NX + i] - v[(j - 1) * NX + i]) / (2.0f * dy)) * 
        ((v[(j + 1) * NX + i] - v[(j - 1) * NX + i]) / (2.0f * dy));
    }
}

__global__ void pressure_poisson_kernel(float *p, float *b, float dx, float dy, int nit) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    for (int it = 0; it < nit; it++) {
        if (i > 0 && i < NX - 1 && j > 0 && j < NY - 1) {
            p[j * NX + i] = ((p[j * NX + i + 1] + p[j * NX + i - 1]) * dy * dy + 
                              (p[(j + 1) * NX + i] + p[(j - 1) * NX + i]) * dx * dx) / 
                             (2 * (dx * dx + dy * dy)) - 
                             dx * dx * dy * dy / (2 * (dx * dx + dy * dy)) * b[j * NX + i];
        }
    }
}

__global__ void update_velocities_kernel(float *u, float *v, float *un, float *vn, 
    float *p, float dx, float dy, float dt, float rho, float nu) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i < NX - 1 && j > 0 && j < NY - 1) {
        // Actualizar u
        u[j * NX + i] = un[j * NX + i] - 
            un[j * NX + i] * dt/dx * (un[j * NX + i] - un[j * NX + i - 1]) - 
            vn[j * NX + i] * dt/dy * (un[j * NX + i] - un[(j - 1) * NX + i]) - 
            dt/(2 * rho * dx) * (p[j * NX + i + 1] - p[j * NX + i - 1]) + 
            nu * (dt/(dx * dx) * (un[j * NX + i + 1] - 2 * un[j * NX + i] + un[j * NX + i - 1]) + 
            dt/(dy*dy) * (un[(j + 1) * NX + i] - 2 * un[j * NX + i] + un[(j - 1) * NX + i]));

        // Actualizar v
        v[j * NX + i] = vn[j * NX + i] - un[j * NX + i] * dt / dx * (vn[j * NX + i] - vn[j * NX + i - 1]) - 
            vn[j * NX + i] * dt / dy * (vn[j * NX + i] - vn[(j - 1) * NX + i]) - 
            dt / (2 * rho * dy) * (p[(j + 1) * NX + i] - p[(j - 1) * NX + i]) + 
            nu * (dt / (dx * dx) * (vn[j * NX + i + 1] - 2 * vn[j * NX + i] + vn[j * NX + i - 1]) + 
            dt / (dy * dy) * (vn[(j + 1) * NX + i] - 2 * vn[j * NX + i] + vn[(j - 1) * NX + i]));

            // Velocidad maxima
            float max_u = 1.0f;
            if(u[j * NX + i] > max_u) {
                u[j * NX + i] = max_u;
            }
            float max_v = 1.0f;
            if(v[j * NX + i] > max_v) {
                v[j * NX + i] = max_v;
            }
    }
}
/*
__global__ void apply_boundaries_kernel(float *u, float *v, float *p, float dx, float dy, float x_center, float y_center, float radius) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // Condiciones de frontera para u y v
    if (i == 0 || i == NX - 1 || j == 0 || j == NY - 1) {
        u[j * NX + i] = 0.0f;  // No movimiento en las paredes
        v[j * NX + i] = 0.0f;  // No movimiento en las paredes
    }

    // Condición de entrada (flujo constante en la frontera izquierda)
    if (i == 0) {
        u[j * NX + i] = 1.0f;  // Flujo constante en la pared izquierda
        v[j * NX + i] = 0.0f;  // Sin velocidad en y
    }

    // Condición de salida libre en la pared derecha
    if (i == NX - 1) {
        p[j * NX + i] = p[j * NX + i - 1];  // dp/dx = 0 en el borde derecho
        u[j * NX + i] = u[j * NX + i - 1];  // Sin gradiente de velocidad en x en la pared derecha
        v[j * NX + i] = v[j * NX + i - 1];  // Sin gradiente de velocidad en y en la pared derecha
    }

    // Calcular la distancia al centro del obstáculo
    float dist = (i * dx - x_center) * (i * dx - x_center) + (j * dy - y_center) * (j * dy - y_center);

    // Verificar si la celda está dentro del obstáculo (círculo)
    if (dist < radius * radius) {
        u[i * NY + j] = 0.0f;  // No hay flujo en el obstáculo en la dirección x
        v[i * NY + j] = 0.0f;  // No hay flujo en el obstáculo en la dirección y
    }
}
*/
__global__ void apply_boundaries_kernel(float *u, float *v, float *p, float dx, float dy) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = i + j * NX;

    // 🔹 Pared en la salida con salida libre
    if (i == NX - 1) {  
        u[idx] = u[idx - 1];  // Sin gradiente de velocidad en x
        v[idx] = v[idx - 1];  // Sin gradiente de velocidad en y
        p[idx] = p[idx - 1];  // Sin gradiente de presión
    }

    // 🔹 Condiciones de frontera periódicas en y
    if (j == 0) {
        int idx_top = i + (NY - 1) * NX;  // Celda superior
        u[idx] = 0;
        v[idx] = 0;
        p[idx] = 0;
    }
    if (j == NY - 1) {
        int idx_bottom = i;  // Celda inferior
        u[idx] = 0;
        v[idx] = 0;
        p[idx] = 0;
    }

    // 🔹 Obstáculo
    const float cx = NX / 4.0f;  // Posición del centro en X
    const float cy = NY / 2.0f;  // Posición del centro en Y
    const float R = NX / 25.0f;  // Radio del obstáculo

    if ((i - cx) * (i - cx) + (j - cy) * (j - cy) * 5 <= R * R) {
        u[idx] = 0.0f;
        v[idx] = 0.0f;
        p[idx] = 0.0f;
    }

    // 🔹 Entrada con velocidad inicial
    if (i == 0) {
        u[idx] = 2.0f;
        v[idx] = 0.0f;
    }
}

__global__ void compute_vorticity_kernel(float *u, float *v, float *vort, float dx, float dy, int nx, int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
        // Calcular la derivada de v con respecto a x
        float dv_dx = (v[j * nx + (i + 1)] - v[j * nx + (i - 1)]) / (2.0f * dx);

        // Calcular la derivada de u con respecto a y
        float du_dy = (u[(j + 1) * nx + i] - u[(j - 1) * nx + i]) / (2.0f * dy);

        // La vorticidad es dv/dx - du/dy
        vort[j * nx + i] = dv_dx - du_dy;
    }
}


int main() {

    // Parámetros físicos
    float rho = 1.0;
    //float nu = 1e-2;
    //float dt = 1e-3;
    const float dt = 0.0002f;

    // Viscosidad cinemática
    const float nu = 0.003f; 
    float dx = 2.0 / (NX -1);
    float dy = 1.0 / (NY - 1);
    int nx = NX, ny = NY;
    int iter_count = 10000;

    float x_center = 256;  // Centro del obstáculo en X
    float y_center = 128;  // Centro del obstáculo en Y
    float radius = 20;    // Radio del obstáculo


    float *u, *v, *p, *b, *vorticity;  // Variables en la memoria del host (CPU)
    float *d_u, *d_v, *d_p, *d_b;  // Variables en la memoria del dispositivo (GPU)

    float *d_un, *d_vn;
    cudaMalloc((void**)&d_un, NX * NY * sizeof(float));
    cudaMalloc((void**)&d_vn, NX * NY * sizeof(float));

    checkCuda(cudaMalloc((void**)&u, NX * NY * sizeof(float)));
    checkCuda(cudaMalloc((void**)&v, NX * NY * sizeof(float)));
    checkCuda(cudaMalloc((void**)&p, NX * NY * sizeof(float)));
    checkCuda(cudaMalloc((void**)&b, NX * NY * sizeof(float)));
    //heckCuda(cudaMalloc((void**)&d_u, NX * NY * sizeof(float)));
    checkCuda(cudaMalloc((void**)&d_v, NX * NY * sizeof(float)));
    checkCuda(cudaMalloc((void**)&d_p, NX * NY * sizeof(float)));
    checkCuda(cudaMalloc((void**)&d_b, NX * NY * sizeof(float)));

    // Copiar los datos desde la CPU hacia la GPU
    cudaMemcpy(d_v, v, NX * NY * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p, p, NX * NY * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, NX * NY * sizeof(float), cudaMemcpyHostToDevice);



    cudaMemset(d_v, 0, NX * NY * sizeof(float));  // Inicializar v en cero
    cudaMemset(d_p, 0, NX * NY * sizeof(float));  // Inicializar p en cero


    
    // Buffer para la visualización de la simulación
    MimirEngine engine;
    engine.init(1920, 1080);

    // Visualización de la velocidad
    MemoryParams u1;
    u1.layout = DataLayout::Layout2D;
    u1.element_count = {(unsigned)(nx), (unsigned)(ny)};
    u1.component_type = ComponentType::Float;
    u1.channel_count = 1;
    u1.resource_type = ResourceType::Buffer;
    auto points_u = engine.createBuffer((void**)&d_u, u1);

    ViewParams view_u1;
    view_u1.element_count = nx * ny;
    view_u1.extent = {(unsigned)(nx), (unsigned)(ny),1};
    view_u1.data_domain = DataDomain::Domain2D;
    view_u1.domain_type = DomainType::Structured;
    view_u1.view_type = ViewType::Voxels;
    view_u1.attributes[AttributeType::Color] = *points_u;
    view_u1.options.default_color = {255,0,0};
    view_u1.options.default_size = 1;

    engine.createView(view_u1);

    MemoryParams v1;
    v1.layout = DataLayout::Layout2D;
    v1.element_count = {(unsigned)(nx), (unsigned)(ny)};
    v1.component_type = ComponentType::Float;
    v1.channel_count = 1;
    v1.resource_type = ResourceType::Buffer;
    auto points_v = engine.createBuffer((void**)&vorticity, v1);

    ViewParams view_v1;
    view_v1.element_count = nx * ny;
    view_v1.extent = {(unsigned)(nx), (unsigned)(ny),1};
    view_v1.data_domain = DataDomain::Domain2D;
    view_v1.domain_type = DomainType::Structured;
    view_v1.view_type = ViewType::Voxels;
    view_v1.attributes[AttributeType::Color] = *points_v;
    view_v1.options.default_color = {0,255,0};
    view_v1.options.default_size = 1;

    engine.createView(view_v1);

    cudaMemset(d_u, 0, NX * NY * sizeof(float));  // Inicializar u en cero


    engine.displayAsync();

    checkCuda(cudaDeviceSynchronize());
/*
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((NX + 15) / 16, (NY + 15) / 16);
*/
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((nx + BLOCK_SIZE - 1) / BLOCK_SIZE, (ny + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cout << "Simulando..." << endl;
    for(size_t i = 0; i < iter_count; i++){

        (i % 10 == 0)?cout << "Iteración " << i << endl: cout << "";

        build_up_b_kernel<<<numBlocks, threadsPerBlock>>>(d_u, d_v, d_b, dx, dy, dt, rho);
        checkCuda(cudaDeviceSynchronize());

        pressure_poisson_kernel<<<numBlocks, threadsPerBlock>>>(d_p, d_b, dx, dy, 50);
        checkCuda(cudaDeviceSynchronize());

        cudaMemcpy(d_un, d_u, NX * NY * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_vn, d_v, NX * NY * sizeof(float), cudaMemcpyDeviceToDevice);

        update_velocities_kernel<<<numBlocks, threadsPerBlock>>>(d_u, d_v, d_un, d_vn, d_p, dx, dy, dt, rho, nu);
        checkCuda(cudaDeviceSynchronize());

        // Calcular la vorticidad
        compute_vorticity_kernel<<<numBlocks, threadsPerBlock>>>(d_u, d_v, vorticity, dx, dy, NX, NY);
        
        // Aplicar condiciones de frontera
        apply_boundaries_kernel<<<numBlocks, threadsPerBlock>>>(d_u, d_v, d_p, dx, dy);
        checkCuda(cudaDeviceSynchronize());

        engine.updateViews();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    engine.showMetrics();
    engine.exit();

    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_p);
    cudaFree(d_b);
    cudaFree(d_un);
    cudaFree(d_vn);


    return 0;
}

