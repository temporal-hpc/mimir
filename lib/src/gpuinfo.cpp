// Device capability + memory-budget helpers (see mimir.hpp). Moved out of the remote-rendering sample
// so any sample can report the GPU and pre-flight a workload without re-deriving the CUDA/NVML tables.
#include "mimir/mimir.hpp"

#include <cuda_runtime_api.h>
#include <nvml.h>

#include <cstdio>
#include <cstring>

namespace mimir
{
namespace
{

// Shader cores per SM by compute capability (NVIDIA's well-known table, from the CUDA samples).
int coresPerSM(int major, int minor)
{
    switch ((major << 4) | minor)
    {
        case 0x30: case 0x32: case 0x35: case 0x37: return 192;  // Kepler
        case 0x50: case 0x52: case 0x53:            return 128;  // Maxwell
        case 0x60:                                  return 64;   // Pascal GP100
        case 0x61: case 0x62:                       return 128;  // Pascal
        case 0x70: case 0x72: case 0x75:            return 64;   // Volta / Turing
        case 0x80:                                  return 64;   // Ampere GA100 (A100)
        case 0x86: case 0x87: case 0x89:            return 128;  // Ampere / Ada
        case 0x90:                                  return 128;  // Hopper
        default:                                    return 128;  // newer archs (approx)
    }
}

// Tensor cores per SM: none before Volta, 8 on Volta/Turing, 4 on Ampere and later.
int tensorPerSM(int major) { return major < 7 ? 0 : (major == 7 ? 8 : 4); }

// RT-core count is NOT queryable via CUDA, so this is a best-effort estimate: ~1 RT core per SM on
// RT-capable GPUs, 0 on datacenter compute parts. Datacenter and consumer archs can share a compute
// capability, so match the known compute-only families by name too.
int rtCores(const cudaDeviceProp& p)
{
    static const char *const compute_only[] = {
        "V100", "A100", "A800", "H100", "H200", "H800", "GH200",
        "B100", "B200", "B300", "GB200", "GB300", // Blackwell datacenter: no RT cores
    };
    for (const char *s : compute_only) { if (std::strstr(p.name, s) != nullptr) { return 0; } }
    const bool cc_datacenter =
        (p.minor == 0 && (p.major == 7 || p.major == 8 || p.major == 9 || p.major == 10));
    const bool rt_capable = (p.major * 10 + p.minor) >= 75 && !cc_datacenter; // Turing+ w/ display
    return rt_capable ? p.multiProcessorCount : 0;
}

// NVENC/NVDEC presence via NVML. NVML ignores CUDA_VISIBLE_DEVICES and enumerates every physical GPU,
// so match the one whose PCI location equals our CUDA device. Encoder-capacity / decoder-utilization
// return NOT_SUPPORTED on a GPU lacking that engine (e.g. the A100: NVDEC yes, NVENC no).
void queryVideoEngines(const cudaDeviceProp& p, bool& nvenc, bool& nvdec)
{
    nvenc = nvdec = false;
    if (nvmlInit_v2() != NVML_SUCCESS) { return; }
    unsigned int count = 0;
    if (nvmlDeviceGetCount_v2(&count) == NVML_SUCCESS)
    {
        for (unsigned int i = 0; i < count; ++i)
        {
            nvmlDevice_t dev{};
            nvmlPciInfo_t pci{};
            if (nvmlDeviceGetHandleByIndex_v2(i, &dev) != NVML_SUCCESS) { continue; }
            if (nvmlDeviceGetPciInfo_v3(dev, &pci) != NVML_SUCCESS)      { continue; }
            if (static_cast<int>(pci.domain) != p.pciDomainID ||
                static_cast<int>(pci.bus)    != p.pciBusID    ||
                static_cast<int>(pci.device) != p.pciDeviceID) { continue; }
            unsigned int cap = 0;
            nvenc = nvmlDeviceGetEncoderCapacity(dev, NVML_ENCODER_QUERY_H264, &cap) == NVML_SUCCESS;
            unsigned int util = 0, period = 0;
            nvdec = nvmlDeviceGetDecoderUtilization(dev, &util, &period) == NVML_SUCCESS;
            break;
        }
    }
    nvmlShutdown();
}

} // namespace

GpuCapabilities queryGpuCapabilities(int device)
{
    GpuCapabilities caps;
    cudaDeviceProp p{};
    if (cudaGetDeviceProperties(&p, device) != cudaSuccess) { return caps; }
    caps.name             = p.name;
    caps.vram_total_bytes = p.totalGlobalMem;
    caps.sm_count         = p.multiProcessorCount;
    caps.cuda_cores       = p.multiProcessorCount * coresPerSM(p.major, p.minor);
    caps.tensor_cores     = p.multiProcessorCount * tensorPerSM(p.major);
    caps.rt_cores         = rtCores(p);
    // Theoretical peak bandwidth = 2 (DDR) * mem clock * bus width / 8; the clock (kHz) and bus (bits)
    // come from device attributes (cudaDeviceProp dropped memoryClockRate in newer CUDA).
    int mem_clock_khz = 0, mem_bus_bits = 0;
    cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, device);
    cudaDeviceGetAttribute(&mem_bus_bits, cudaDevAttrGlobalMemoryBusWidth, device);
    caps.mem_bandwidth_gbps =
        2.0 * static_cast<double>(mem_clock_khz) * 1e3 * (static_cast<double>(mem_bus_bits) / 8.0) / 1e9;
    queryVideoEngines(p, caps.nvenc, caps.nvdec);
    return caps;
}

std::string gpuBanner(int device, const GpuCapabilities& c)
{
    char buf[512];
    std::snprintf(buf, sizeof(buf),
        "device %d (%s) | %.0f GB | %.0f GB/s mem BW | %d SMs | %d CUDA cores | %d tensor cores | "
        "%d RT cores (%s) | NVENC %s | NVDEC %s",
        device, c.name.c_str(),
        static_cast<double>(c.vram_total_bytes) / (1024.0 * 1024.0 * 1024.0), c.mem_bandwidth_gbps,
        c.sm_count, c.cuda_cores, c.tensor_cores,
        c.rt_cores, c.rt_cores > 0 ? "hardware BVH traversal" : "none -> software BVH",
        c.nvenc ? "yes" : "no", c.nvdec ? "yes" : "no");
    return buf;
}

uint64_t interopBytesPerParticle(LightModel light_model, bool lod_active)
{
    const bool pt_no_lod = (light_model == LightModel::PathTracing) && !lod_active;
    return 12ull + (pt_no_lod ? 24ull : 0ull);
}

MemoryBudget memoryBudget(uint64_t particle_count, uint64_t bytes_per_particle, int device)
{
    MemoryBudget b;
    if (bytes_per_particle == 0) { bytes_per_particle = 1; }
    int prev = 0; cudaGetDevice(&prev);
    if (device != prev) { cudaSetDevice(device); }
    cudaMemGetInfo(&b.free_bytes, &b.total_bytes);
    if (device != prev) { cudaSetDevice(prev); }
    b.max_particles = b.free_bytes / bytes_per_particle;
    b.fits          = (particle_count <= b.max_particles);
    return b;
}

} // namespace mimir
