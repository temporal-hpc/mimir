#include <experimental/source_location> // std::source_location
#include <curand_kernel.h>
#include <chrono> // std::chrono
#include <string> // std::stoul

#include "nvmlPower.hpp"
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>

using chrono_tp = std::chrono::time_point<std::chrono::high_resolution_clock>;
using source_location = std::experimental::source_location;

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

__global__ void initSystem(float *coords, size_t point_count,
    curandState *global_states, uint3 extent, unsigned seed)
{
    auto points = reinterpret_cast<float4*>(coords);
    auto tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx < point_count)
    {
        auto local_state = global_states[tidx];
        curand_init(seed, tidx, 0, &local_state);
        auto rx = extent.x * curand_uniform(&local_state);
        auto ry = extent.y * curand_uniform(&local_state);
        auto rz = extent.z * curand_uniform(&local_state);
        points[tidx] = {rx, ry, rz, 0};
        global_states[tidx] = local_state;
    }
}

__global__ void integrate3d(float *coords, size_t point_count,
    curandState *global_states, uint3 extent)
{
    auto points = reinterpret_cast<float4*>(coords);
    auto tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx < point_count)
    {
        auto local_state = global_states[tidx];
        auto p = points[tidx];
        p.x += curand_normal(&local_state);
        if (p.x > extent.x) p.x = extent.x;
        if (p.x < 0) p.x = 0;
        p.y += curand_normal(&local_state);
        if (p.y > extent.x) p.y = extent.y;
        if (p.y < 0) p.y = 0;
        p.z += curand_normal(&local_state);
        if (p.z > extent.z) p.z = extent.z;
        if (p.z < 0) p.z = 0;
        points[tidx] = p;
        global_states[tidx] = local_state;
    }
}

int main(int argc, char *argv[])
{
    float *d_coords       = nullptr;
    curandState *d_states = nullptr;
    unsigned block_size   = 256;
    unsigned seed         = 123456;
    uint3 extent          = {200, 200, 200};

    // Default values for this program
    unsigned int point_count = 100;
    int iter_count = 10000;
    if (argc >= 2) point_count = std::stoul(argv[1]);
    if (argc >= 3) iter_count = std::stoi(argv[2]);

    checkCuda(cudaMalloc((void**)&d_coords, sizeof(float4) * point_count));
    checkCuda(cudaMalloc(&d_states, sizeof(curandState) * point_count));
    unsigned grid_size = (point_count + block_size - 1) / block_size;
    initSystem<<<grid_size, block_size>>>(d_coords, point_count, d_states, extent, seed);
    checkCuda(cudaDeviceSynchronize());

    // Initialize point cloud structure on host memory
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    //pcl::PointCloud<pcl::PointXYZ> cloud;
    cloud->width = point_count;
    cloud->height = 1;
    cloud->is_dense = true;
    cloud->resize(point_count);
    checkCuda(cudaMemcpy(cloud->points.data(), d_coords,
        sizeof(float4) * point_count, cudaMemcpyDeviceToHost)
    );

    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer ("3D Viewer", true));
    viewer->setBackgroundColor (0.0, 0.0, 0.0);
    viewer->addPointCloud<pcl::PointXYZ> (cloud, "points3d");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "points3d");
    viewer->addCoordinateSystem (1.0);
    viewer->initCameraParameters ();
    viewer->setSize(1920, 1080);
    viewer->setCameraPosition(-199.419, 277.404, 279.833, 56.991, 13.6156, 47.422, 0, 0, 1);
    //viewer->setFullScreen(true); // Produces segmentation fault

    // Main loop variables
    int iter_idx = 0;
    float total_graphics_time = 0;
    chrono_tp last_time = {};
    std::array<float,240> frame_times{};
    size_t total_frame_count = 0;

    GPUPowerBegin("gpu", 100);
    while (!viewer->wasStopped())
    {
        if (iter_idx < iter_count)
        {
            // Measure frame time
            static chrono_tp start_time = std::chrono::high_resolution_clock::now();
            chrono_tp current_time = std::chrono::high_resolution_clock::now();
            if (iter_idx == 0)
            {
                last_time = start_time;
            }
            float frame_time = std::chrono::duration<float, std::chrono::seconds::period>(current_time - last_time).count();

            viewer->removeAllPointClouds();

            integrate3d<<<grid_size, block_size>>>(d_coords, point_count, d_states, extent);
            checkCuda(cudaDeviceSynchronize());

            checkCuda(cudaMemcpy(cloud->points.data(), d_coords,
                sizeof(float4) * point_count, cudaMemcpyDeviceToHost)
            );

            viewer->addPointCloud(cloud, "points3d");
            viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "points3d");
            viewer->spinOnce(100);

            iter_idx++;
            total_frame_count++;
            total_graphics_time += frame_time;
            frame_times[iter_idx % frame_times.size()] = frame_time;
            last_time = current_time;
        } else {
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

            printf("PCL,FHD,%d,%f,%f,%f,%f,%f,%f,%f,%f\n",
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

            viewer->close(); // This causes segfault :(
        }
        /*std::vector<pcl::visualization::Camera> cam;
        viewer->getCameras(cam);
        printf("pos:   %f %f %f\nview:  %f %f %f\nfocal: %f %f %f\n",
            cam[0].pos[0], cam[0].pos[1], cam[0].pos[2],
            cam[0].view[0], cam[0].view[1], cam[0].view[2],
            cam[0].focal[0], cam[0].focal[1], cam[0].focal[2]
        );*/
    }

    return EXIT_SUCCESS;
}
