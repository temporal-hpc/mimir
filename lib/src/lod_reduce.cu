// LodReduce implementation: custom-CUDA atomic scatter-into-accumulator + compacting emit. See
// lod_reduce.hpp for the public contract and .superpowers/sdd/task-1-brief.md for the design.
//
// Pipeline (all async on the caller's stream; the N^3 accumulator is allocated once in the ctor):
//   1. clearKernel:   zero counts[N^3] (and sums[3*N^3] when centroid).
//   2. scatterKernel: size_t grid-stride loop over `count` particles -- for each particle, bins it
//      into its cell and either (centroid) atomically accumulates a u32 count + int64 fixed-point
//      position sum, or (cell-center) writes a benign non-atomic occupancy flag (every writer
//      stores the identical value 1u, so the race is harmless).
//   3. emitKernel:    one thread per N^3 cell; occupied cells (counts[c] != 0) atomically claim a
//      compaction slot (atomicAdd on a single global counter) and write their representative
//      position (mass centroid or cell geometric center) into that slot of reduced_pos.
//
// NO CUB/Thrust: the scatter loop is indexed with size_t/uint64_t (never int), so it scales past
// 2^32 particles -- the entire point of this rewrite vs. the retired CUB implementation (whose
// Encode/ReduceByKey item counts capped at INT32_MAX).
//
// Determinism: the occupied-cell SET is deterministic (a cell is occupied iff >=1 particle maps to
// it, independent of scatter order). Centroid sums use atomicAdd on integer fixed-point values
// (scale 2^30), so they are also order-independent and thus deterministic run-to-run -- floating
// point atomicAdd would not have this property (its result depends on summation order).

#include "mimir/lod_reduce.hpp"

#include <cuda_runtime.h>

#include <cassert>
#include <cstdio>
#include <cstdlib>

namespace mimir
{
namespace
{

void checkCuda(cudaError_t err, const char* what)
{
    if (err != cudaSuccess)
    {
        std::fprintf(stderr, "LodReduce CUDA error in %s: %s\n", what, cudaGetErrorString(err));
        std::abort();
    }
}

// Checks both the deferred launch-configuration error (cudaGetLastError, cleared as a side effect)
// and any error already latched from a prior async call on this stream (cudaPeekAtLastError,
// which does NOT clear it -- so a real error is not silently swallowed here).
void checkLaunch(const char* what)
{
    checkCuda(cudaGetLastError(), what);
    checkCuda(cudaPeekAtLastError(), what);
}

// Number of cells (N^3) as a 64-bit value to avoid any 32-bit overflow while computing it (the
// caller-enforced cap gridN <= 1625 keeps the RESULT itself just under 2^32).
uint64_t numCells(uint32_t gridN)
{
    return static_cast<uint64_t>(gridN) * gridN * gridN;
}

constexpr int kThreads = 256;
constexpr unsigned long long kFixedScale = 1ull << 30; // 2^30, matches pathtrace_lod_scatter.slang

inline uint32_t gridFor(uint64_t n, int threads = kThreads)
{
    uint64_t blocks = (n + static_cast<uint64_t>(threads) - 1) / static_cast<uint64_t>(threads);
    // Clamp to the max 1-D grid dimension (2^31-1); the scatter/clear/emit kernels use a
    // size_t/uint64_t grid-stride loop internally so under-launching here just means each thread
    // does more iterations -- never a correctness issue, only a (harmless) launch-size cap.
    constexpr uint64_t kMaxBlocks = 2147483647ull;
    if (blocks > kMaxBlocks) blocks = kMaxBlocks;
    if (blocks < 1) blocks = 1;
    return static_cast<uint32_t>(blocks);
}

// Cell index per axis: clamp(int((p+1)*0.5*N), 0, N-1). Matches pathtrace_lod_scatter.slang.
static __device__ __forceinline__ uint32_t cellId(float px, float py, float pz, uint32_t N)
{
    float n = static_cast<float>(N);
    int cx = static_cast<int>(fminf(fmaxf((px + 1.f) * 0.5f * n, 0.f), n - 1.f));
    int cy = static_cast<int>(fminf(fmaxf((py + 1.f) * 0.5f * n, 0.f), n - 1.f));
    int cz = static_cast<int>(fminf(fmaxf((pz + 1.f) * 0.5f * n, 0.f), n - 1.f));
    return static_cast<uint32_t>(cx) + N * (static_cast<uint32_t>(cy) + N * static_cast<uint32_t>(cz));
}

// Fixed-point encode: q = (unsigned long long)((clamp(p,-1,1)+1)*0.5*2^30). Matches the slang.
static __device__ __forceinline__ unsigned long long fixedPoint(float p)
{
    float c = fminf(fmaxf(p, -1.f), 1.f);
    double v = (static_cast<double>(c) + 1.0) * 0.5 * static_cast<double>(kFixedScale);
    return static_cast<unsigned long long>(v);
}

// Zero counts[N^3] (and sums[3*N^3] when centroid). Grid-stride over the N^3 cells with size_t
// indexing so it scales to any gridN <= 1625 (N^3 < 2^32, but the loop itself uses uint64_t).
__global__ void clearKernel(uint32_t* counts, unsigned long long* sums, uint64_t nCells,
                             bool centroid)
{
    uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
    for (uint64_t c = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; c < nCells;
         c += stride)
    {
        counts[c] = 0u;
        if (centroid)
        {
            sums[3 * c + 0] = 0ull;
            sums[3 * c + 1] = 0ull;
            sums[3 * c + 2] = 0ull;
        }
    }
}

// Scatters every particle into the N^3 accumulator. size_t/uint64_t grid-stride loop over `count`
// particles -- NEVER `int` -- so this scales past 2^32 particles (the whole point of this
// module vs. CUB/Thrust, whose item counts are int-sized).
//   centroid  : atomicAdd(&counts[c], 1) + 3x atomicAdd(&sums[3c+k], fixedPoint(pos_k)) (u64).
//   cell-cent : counts[c] = 1u, a benign non-atomic store -- every writer to a given cell writes
//               the identical value, so concurrent writes race harmlessly.
__global__ void scatterKernel(const float* pos, uint64_t count, uint32_t N, bool centroid,
                               uint32_t* counts, unsigned long long* sums)
{
    uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
    for (uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += stride)
    {
        float px = pos[i * 3 + 0];
        float py = pos[i * 3 + 1];
        float pz = pos[i * 3 + 2];
        uint32_t c = cellId(px, py, pz, N);

        if (centroid)
        {
            atomicAdd(&counts[c], 1u);
            atomicAdd(&sums[3 * static_cast<uint64_t>(c) + 0], fixedPoint(px));
            atomicAdd(&sums[3 * static_cast<uint64_t>(c) + 1], fixedPoint(py));
            atomicAdd(&sums[3 * static_cast<uint64_t>(c) + 2], fixedPoint(pz));
        }
        else
        {
            counts[c] = 1u;
        }
    }
}

// One thread per N^3 cell (grid-stride, size_t indexing): occupied cells (counts[c] != 0) claim a
// compaction slot via a single global atomicAdd(occupied, 1) and write their representative
// position into that slot of reduced_pos.
//   cell-cent: center = -1 + (cell + 0.5) * (2/N) per axis.
//   centroid : de-fixedpoint sums[3c+k]/counts[c] back to [-1,1].
__global__ void emitKernel(const uint32_t* counts, const unsigned long long* sums, uint64_t nCells,
                            uint32_t N, bool centroid, float* reduced_pos, uint32_t* occupied)
{
    uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
    for (uint64_t c = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; c < nCells;
         c += stride)
    {
        uint32_t cnt = counts[c];
        if (cnt == 0u) continue;

        float3 rep;
        if (centroid)
        {
            double invCount = 1.0 / static_cast<double>(cnt);
            double fx = static_cast<double>(sums[3 * c + 0]) * invCount;
            double fy = static_cast<double>(sums[3 * c + 1]) * invCount;
            double fz = static_cast<double>(sums[3 * c + 2]) * invCount;
            double scale = static_cast<double>(kFixedScale);
            rep = make_float3(static_cast<float>((fx / scale) * 2.0 - 1.0),
                               static_cast<float>((fy / scale) * 2.0 - 1.0),
                               static_cast<float>((fz / scale) * 2.0 - 1.0));
        }
        else
        {
            uint32_t lin = static_cast<uint32_t>(c);
            uint32_t cx = lin % N;
            uint32_t cy = (lin / N) % N;
            uint32_t cz = lin / (N * N);
            float cs = 2.0f / static_cast<float>(N);
            rep = make_float3(-1.0f + (static_cast<float>(cx) + 0.5f) * cs,
                               -1.0f + (static_cast<float>(cy) + 0.5f) * cs,
                               -1.0f + (static_cast<float>(cz) + 0.5f) * cs);
        }

        uint32_t slot = atomicAdd(occupied, 1u);
        // 64-bit base: slot can approach N^3 (< 2^32), so slot*3 must be computed in 64 bits or it
        // wraps past ~1.43 B occupied cells and scatters representatives to garbage low offsets. The
        // slang emit widens the same way (uint64_t(slot) * 12ull). Mirrors the AABB-path 64-bit fix.
        size_t base = static_cast<size_t>(slot) * 3;
        reduced_pos[base + 0] = rep.x;
        reduced_pos[base + 1] = rep.y;
        reduced_pos[base + 2] = rep.z;
    }
}

} // namespace

struct LodReduce::Impl
{
    uint64_t max_particles;
    uint32_t gridN;
    bool centroid;
    uint64_t nCells;

    uint32_t* counts = nullptr;             // uint32_t[nCells]
    unsigned long long* sums = nullptr;     // uint64_t[3*nCells]  (centroid only)

    Impl(uint64_t max_particles_, uint32_t gridN_, bool centroid_)
        : max_particles(max_particles_), gridN(gridN_), centroid(centroid_),
          nCells(numCells(gridN_))
    {
        checkCuda(cudaMalloc(&counts, nCells * sizeof(uint32_t)), "cudaMalloc counts");
        if (centroid)
        {
            checkCuda(cudaMalloc(&sums, 3 * nCells * sizeof(unsigned long long)),
                      "cudaMalloc sums");
        }
    }

    ~Impl()
    {
        cudaFree(counts);
        cudaFree(sums);
    }

    void reduce(cudaStream_t stream, const void* positions_dev, uint64_t count,
                void* reduced_pos_dev, uint32_t* occupied_dev)
    {
        const float* positions = static_cast<const float*>(positions_dev);
        float* reduced_pos = static_cast<float*>(reduced_pos_dev);

        // positions_dev is sized for max_particles; a larger count would over-read it in scatterKernel.
        assert(count <= max_particles && "LodReduce::reduce count exceeds the configured max_particles");

        checkCuda(cudaMemsetAsync(occupied_dev, 0, sizeof(uint32_t), stream),
                  "memsetAsync occupied");

        // Per-kernel timing (MIMIR_LOD_KTIME): record stream events around clear/scatter/emit so one
        // run reports which kernel dominates -- the read is ~1.5 ms of 11.4 GB at 7.6 TB/s, so a large
        // total points at the scattered accumulator writes, not the read. Events are on the stream (no
        // inter-kernel serialization); the elapsed times are read after a sync at the end. Throttled.
        static const bool ktime = (std::getenv("MIMIR_LOD_KTIME") != nullptr);
        static uint64_t ktime_call = 0;
        const bool ktime_now = ktime && ((ktime_call++ % 30) == 0);
        cudaEvent_t e_start{}, e_clear{}, e_scatter{}, e_emit{};
        if (ktime_now)
        {
            cudaEventCreate(&e_start); cudaEventCreate(&e_clear);
            cudaEventCreate(&e_scatter); cudaEventCreate(&e_emit);
            cudaEventRecord(e_start, stream);
        }

        clearKernel<<<gridFor(nCells), kThreads, 0, stream>>>(counts, sums, nCells, centroid);
        checkLaunch("clearKernel");
        if (ktime_now) { cudaEventRecord(e_clear, stream); }

        if (count > 0)
        {
            scatterKernel<<<gridFor(count), kThreads, 0, stream>>>(
                positions, count, gridN, centroid, counts, sums);
            checkLaunch("scatterKernel");
        }
        if (ktime_now) { cudaEventRecord(e_scatter, stream); }

        emitKernel<<<gridFor(nCells), kThreads, 0, stream>>>(
            counts, sums, nCells, gridN, centroid, reduced_pos, occupied_dev);
        checkLaunch("emitKernel");

        if (ktime_now)
        {
            cudaEventRecord(e_emit, stream);
            cudaEventSynchronize(e_emit);
            float t_clear = 0.f, t_scatter = 0.f, t_emit = 0.f;
            cudaEventElapsedTime(&t_clear,   e_start,   e_clear);
            cudaEventElapsedTime(&t_scatter, e_clear,   e_scatter);
            cudaEventElapsedTime(&t_emit,    e_scatter, e_emit);
            std::fprintf(stderr,
                "[lod-ktime] N=%llu cells=%llu %s | clear %.2f ms | scatter %.2f ms | emit %.2f ms | total %.2f ms\n",
                static_cast<unsigned long long>(count),
                static_cast<unsigned long long>(nCells),
                centroid ? "centroid" : "cell-center",
                t_clear, t_scatter, t_emit, t_clear + t_scatter + t_emit);
            cudaEventDestroy(e_start); cudaEventDestroy(e_clear);
            cudaEventDestroy(e_scatter); cudaEventDestroy(e_emit);
        }
    }
};

LodReduce::LodReduce(uint64_t max_particles, uint32_t gridN, bool centroid)
    : impl(new Impl(max_particles, gridN, centroid))
{
}

LodReduce::~LodReduce()
{
    delete impl;
}

size_t LodReduce::accumulatorBytes(uint32_t gridN, bool centroid)
{
    uint64_t nCells = numCells(gridN);
    size_t bytes = nCells * sizeof(uint32_t);
    if (centroid)
    {
        bytes += 3 * nCells * sizeof(unsigned long long);
    }
    return bytes;
}

void LodReduce::reduce(cudaStream_t stream, const void* positions_dev, uint64_t count,
                        void* reduced_pos_dev, uint32_t* occupied_dev)
{
    impl->reduce(stream, positions_dev, count, reduced_pos_dev, occupied_dev);
}

} // namespace mimir
