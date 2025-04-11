#include <experimental/source_location> // std::source_location
#include <curand_kernel.h>
#include <chrono> // std::chrono
#include <string> // std::stoul
#include <cmath>
#include <iostream>

#include "nvmlPower.hpp"

#include "datoviz.h"
#include "datoviz_protocol.h"

using chrono_tp = std::chrono::time_point<std::chrono::high_resolution_clock>;
using source_location = std::experimental::source_location;

// Simulation variables
DvzApp *app;
DvzScene *scene;
DvzPanel *panel;
DvzVisual *visual;
std::vector<float3> h_pos;
float *d_coords         = nullptr;
curandState *d_states = nullptr;
unsigned int point_count = 100;
int iter_count = 1000;
unsigned seed         = 123456;
uint3 extent          = {1, 1, 1};
unsigned int block_size = 256;
unsigned int grid_size;
int iter_idx = 0;
float total_graphics_time = 0;
chrono_tp last_time = {};
std::array<float,240> frame_times{};
size_t total_frame_count = 0;

constexpr void checkCuda(cudaError_t code, bool panic = true,
    source_location src = source_location::current())
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "CUDA assertion: %s in function %s at %s(%d)\n",
            cudaGetErrorString(code), src.function_name(), src.file_name(), src.line()
        );
        if (panic)
        {
            exit(EXIT_FAILURE);
        }
    }
}

// Init RNG states
__global__ void initRng(curandState *states, unsigned int rng_count, unsigned int seed)
{
    auto tidx = blockDim.x * blockIdx.x + threadIdx.x;
    curand_init(seed, tidx, 0, &states[tidx]);
}

// Init starting positions
__global__ void initPos(float *coords, size_t point_count, curandState *rng)
{
    auto points = reinterpret_cast<float3*>(coords);
    auto tidx = blockDim.x * blockIdx.x + threadIdx.x;
    auto stride = gridDim.x * blockDim.x;
    auto state = rng[tidx];
    for (int i = tidx; i < point_count; i += stride)
    {
        auto rx = curand_uniform(&state);
        auto ry = curand_uniform(&state);
        auto rz = curand_uniform(&state);
        points[i] = {rx, ry, rz};
    }
    rng[tidx] = state;
}

__global__ void integrate3d(float *coords, size_t point_count, curandState *rng)
{
    auto points = reinterpret_cast<float3*>(coords);
    auto tidx = blockDim.x * blockIdx.x + threadIdx.x;
    auto stride = gridDim.x * blockDim.x;
    auto local_state = rng[tidx];
    for (int i = tidx; i < point_count; i += stride)
    {
        // Read current position
        auto p = points[i];

        // Generate random displacements with device RNG state
        p.x += 0.1 * curand_normal(&local_state);
        p.y += 0.1 * curand_normal(&local_state);
        p.z += 0.1 * curand_normal(&local_state);

        // Correct positions to bounds
        if (p.x > 1.f) p.x = 1.f;
        if (p.x < 0.f) p.x = 0.f;
        if (p.y > 1.f) p.y = 1.f;
        if (p.y < 0.f) p.y = 0.f;
        if (p.z > 1.f) p.z = 1.f;
        if (p.z < 0.f) p.z = 0.f;

        // Write updated position
        points[i] = p;
    }
    rng[tidx] = local_state;
}

void callback(DvzApp* app, DvzId window_id, DvzTimerEvent ev)
{
    // Measure frame time
    static chrono_tp start_time = std::chrono::high_resolution_clock::now();
    chrono_tp current_time = std::chrono::high_resolution_clock::now();
    if (iter_idx == 0)
    {
        last_time = start_time;
    }
    float frame_time = std::chrono::duration<float, std::chrono::seconds::period>(current_time - last_time).count();

    integrate3d<<<grid_size, block_size>>>(d_coords, point_count, d_states);
    checkCuda(cudaDeviceSynchronize());

    checkCuda(cudaMemcpy(h_pos.data(), d_coords,
        sizeof(vec3) * point_count, cudaMemcpyDeviceToHost)
    );
    checkCuda(cudaDeviceSynchronize());
    dvz_sphere_position(visual, 0, point_count, reinterpret_cast<vec3*>(h_pos.data()), 0);

    iter_idx++;
    total_frame_count++;
    total_graphics_time += frame_time;
    frame_times[iter_idx % frame_times.size()] = frame_time;
    last_time = current_time;

    if (iter_idx == iter_count)
    {
        auto frame_sample_size = std::min(frame_times.size(), total_frame_count);
        float total_frame_time = 0;
        for (size_t i = 0; i < frame_sample_size; ++i) total_frame_time += frame_times[i];
        auto framerate = frame_times.size() / total_frame_time;

        // Nvml memory report
        nvmlMemory_v2_t meminfo;
        meminfo.version = (unsigned int)(sizeof(nvmlMemory_v2_t) | (2 << 24U));
        nvmlDeviceGetMemoryInfo_v2(getNvmlDevice(), &meminfo);

        constexpr double gigabyte = 1024.0 * 1024.0 * 1024.0;
        double nvml_free = meminfo.free / gigabyte;
        double nvml_reserved = meminfo.reserved / gigabyte;
        double nvml_total = meminfo.total / gigabyte;
        double nvml_used = meminfo.used / gigabyte;

        auto gpu = GPUPowerEnd();

        checkCuda(cudaFree(d_states));
        checkCuda(cudaFree(d_coords));

        printf("Datoviz,FHD,%d,%f,%f,%f,%f,%f,%f,%f,%f\n",
            point_count,
            framerate,
            gpu.average_power,
            gpu.total_energy,
            gpu.total_time,
            nvml_free,
            nvml_reserved,
            nvml_total,
            nvml_used
        );

        // Flush output before segmentation fault
        std::flush(std::cout);

        // This causes segfault :(
        dvz_scene_destroy(scene);
        dvz_app_destroy(app);
    }
}

int main(int argc, char *argv[])
{
    // Sanity check
    assert(sizeof(vec3) == sizeof(float3));

    // Default values for this program
    int width      = 1920;
    int height     = 1080;
    if (argc >= 2) point_count = std::stoul(argv[1]);
    if (argc >= 3) iter_count = std::stoi(argv[2]);

    // Set manually CUDA device to 0; change if needed
    const int device_id = 0;
    checkCuda(cudaSetDevice(device_id));

    // Retrieve number of streaming multiprocessors (SMs)
    int sm_count = -1;
    checkCuda(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id));

    // Retrieve max number of threads per SM
    int max_sm_thread_count = -1;
    checkCuda(cudaDeviceGetAttribute(
        &max_sm_thread_count, cudaDevAttrMaxThreadsPerMultiProcessor, device_id)
    );

    // Determine kernel parameters from previous values
    const int max_block_count = max_sm_thread_count / block_size;
    const int rng_state_count = max_block_count * sm_count * block_size;
    grid_size = (rng_state_count + block_size - 1) / block_size;

    checkCuda(cudaMalloc((void**)&d_coords, sizeof(float3) * point_count));
    checkCuda(cudaMalloc(&d_states, sizeof(curandState) * rng_state_count));
    initRng<<<grid_size, block_size>>>(d_states, rng_state_count, seed);
    checkCuda(cudaDeviceSynchronize());
    initPos<<<grid_size, block_size>>>(d_coords, point_count, d_states);
    checkCuda(cudaDeviceSynchronize());

    app = dvz_app(0);
    auto batch = dvz_app_batch(app);
    scene = dvz_scene(batch);
    auto figure = dvz_figure(scene, width, height, 0);

    panel = dvz_panel_default(figure);
    auto arcball = dvz_panel_arcball(panel);
    visual = dvz_sphere(batch, 0);
    dvz_sphere_alloc(visual, point_count);

    // auto fig_id = dvz_figure_id(figure);
    // cvec4 bg_color(127, 127, 127, 255);
    // dvz_set_background(batch, fig_id, bg_color);
    // dvz_panel_update(panel);

    h_pos.resize(point_count);
    checkCuda(cudaMemcpy(h_pos.data(), d_coords, sizeof(vec3) * point_count, cudaMemcpyDeviceToHost));

    std::vector<uchar4> h_color(point_count, {255,255,255,255});
    std::vector<float> h_size(point_count, 5.f);

    dvz_sphere_position(visual, 0, point_count, reinterpret_cast<vec3*>(h_pos.data()), 0);
    dvz_sphere_color(visual, 0, point_count, reinterpret_cast<cvec4*>(h_color.data()), 0);
    dvz_sphere_size(visual, 0, point_count, h_size.data(), 0);

    // Light position.
    vec3 light_pos(-5, +5, +100);
    dvz_sphere_light_pos(visual, light_pos);
    // Light parameters.
    vec4 light_params(.4, .8, 2, 32);
    dvz_sphere_light_params(visual, light_params);

    dvz_panel_visual(panel, visual, 0);
    vec3 arcball_init(.5f, 0.f, 0.f);
    dvz_arcball_initial(arcball, arcball_init);
    dvz_panel_update(panel);

    dvz_app_timer(app, 0, 1.0 / 60.0, iter_count);
    dvz_app_ontimer(app, callback, nullptr);

    GPUPowerBegin("gpu", 100);
    dvz_scene_run(scene, app, 0);

    return EXIT_SUCCESS;
}

