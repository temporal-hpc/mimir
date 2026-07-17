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
// particles -- NEVER `int` -- so this scales past 2^32 (the whole point vs. CUB/Thrust, whose item
// counts are int-sized). The per-thread AoS float3 read (3 stride-3 loads) is already cache-line
// efficient on Blackwell -- staging the tile through shared memory was measured SLOWER (the syncthreads
// + shared round-trip outweighed any coalescing gain), so we keep the direct read.
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
            // Cell-center only needs an occupancy flag, and each occupied cell is hit by MANY particles
            // (e.g. ~1392:1), all writing the identical 1. Skip the store when the cell is already
            // marked: this turns ~all of the 1e9 scattered writes into cheap (L2-resident) reads plus
            // only ~one write per occupied cell, cutting the write traffic that dominates scatter. The
            // read/write race stays benign -- every writer stores the same value 1.
            if (counts[c] == 0u) { counts[c] = 1u; }
        }
    }
}

constexpr int kWarp = 32;

// Block-wide exclusive prefix scan of a per-thread predicate, via warp shuffle -- no serialized
// atomics. Returns this thread's exclusive rank among the block's set threads and, via blockTotal, the
// block's set count. Intra-warp rank is __ballot_sync + __popc; the per-warp totals are exclusive-
// scanned by warp 0 with __shfl_up_sync. s_scan needs (blockDim.x/32 + 1) uint32_t. blockDim.x must be
// a multiple of 32. All threads of the block must call this (uniform control flow for __syncthreads).
__device__ __forceinline__ uint32_t blockExclusiveScan(bool pred, uint32_t* s_scan, uint32_t& blockTotal)
{
    const int lane = threadIdx.x & (kWarp - 1);
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int warpsPerBlock = static_cast<int>(blockDim.x) >> 5;

    const unsigned active = __ballot_sync(0xffffffffu, pred);
    const uint32_t laneRank = __popc(active & ((1u << lane) - 1u)); // exclusive prefix within the warp
    if (lane == 0) { s_scan[warp] = __popc(active); }              // per-warp set count
    __syncthreads();

    if (warp == 0)                                                 // exclusive-scan the warp counts
    {
        uint32_t v = (lane < warpsPerBlock) ? s_scan[lane] : 0u;
        const uint32_t self = v;
        for (int o = 1; o < kWarp; o <<= 1)
        {
            const uint32_t n = __shfl_up_sync(0xffffffffu, v, o);
            if (lane >= o) { v += n; }
        }
        if (lane < warpsPerBlock)      { s_scan[lane] = v - self; }   // exclusive per-warp offset
        if (lane == warpsPerBlock - 1) { s_scan[warpsPerBlock] = v; } // block total (inclusive end)
    }
    __syncthreads();

    blockTotal = s_scan[warpsPerBlock];
    return s_scan[warp] + laneRank;
}

// Representative position of an occupied cell c: mass centroid (de-fixedpoint sums/count) or the cell's
// geometric center. Caller guarantees c < nCells and counts[c] != 0.
__device__ __forceinline__ float3 cellRepresentative(uint64_t c, uint32_t N, bool centroid,
    const uint32_t* counts, const unsigned long long* sums)
{
    if (centroid)
    {
        double invCount = 1.0 / static_cast<double>(counts[c]);
        double fx = static_cast<double>(sums[3 * c + 0]) * invCount;
        double fy = static_cast<double>(sums[3 * c + 1]) * invCount;
        double fz = static_cast<double>(sums[3 * c + 2]) * invCount;
        double scale = static_cast<double>(kFixedScale);
        return make_float3(static_cast<float>((fx / scale) * 2.0 - 1.0),
                           static_cast<float>((fy / scale) * 2.0 - 1.0),
                           static_cast<float>((fz / scale) * 2.0 - 1.0));
    }
    uint32_t lin = static_cast<uint32_t>(c);
    uint32_t cx = lin % N;
    uint32_t cy = (lin / N) % N;
    uint32_t cz = lin / (N * N);
    float cs = 2.0f / static_cast<float>(N);
    return make_float3(-1.0f + (static_cast<float>(cx) + 0.5f) * cs,
                       -1.0f + (static_cast<float>(cy) + 0.5f) * cs,
                       -1.0f + (static_cast<float>(cz) + 0.5f) * cs);
}

// Fully atomic-free compacting emit, in three passes over a PERSISTENT grid (one block per resident
// slot, so gridDim is small -- a few thousand). Earlier designs claimed each cell's (or each block's)
// slot with a global atomicAdd; on a distributed/multi-die L2 even a few thousand single-address
// atomics cost ~1 us each (the B300 emit was atomic-latency-bound). Here NO kernel touches a shared
// global counter:
//   1. emitCountKernel: each block counts its emitters (block-wide warp-shuffle scan, summed over its
//      strided cells) and stores the count in block_counts[blockIdx].
//   2. scanBlockCountsKernel: exclusive-scans block_counts -> per-block base offsets (in place), and
//      writes the grand total (= occupied-cell count) to *occupied. One small block, no atomics.
//   3. emitWriteKernel: each block reads its base offset and re-scans/writes its emitters contiguously.
// The count/write kernels MUST launch with the same grid so blockIdx indexes block_counts consistently.
//   cell-center: center = -1 + (cell + 0.5) * (2/N) per axis.  centroid: mass centroid in [-1,1].

// Pass 1 -- per-block emitter count (no writes to reduced_pos, no atomics).
__global__ void emitCountKernel(const uint32_t* counts, uint64_t nCells, uint32_t* block_counts)
{
    __shared__ uint32_t s_scan[kWarp + 1];
    const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
    const uint64_t start  = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t iters_end = ((nCells + stride - 1) / stride) * stride;

    uint32_t blockOwn = 0;
    for (uint64_t c = start; c < iters_end; c += stride)
    {
        const bool emit = (c < nCells) && (counts[c] != 0u);
        uint32_t total = 0;
        blockExclusiveScan(emit, s_scan, total);  // uniform call; we use only the block total here
        if (threadIdx.x == 0) { blockOwn += total; }
        __syncthreads();  // s_scan reused next iteration
    }
    if (threadIdx.x == 0) { block_counts[blockIdx.x] = blockOwn; }
}

// Pass 2 -- exclusive-scan the (small) per-block counts into base offsets, in place; write grand total.
// Single block; block_counts has one entry per emit block (a few thousand, bounded by kMaxScanElems).
constexpr uint32_t kMaxScanElems = 8192;
__global__ void scanBlockCountsKernel(uint32_t* block_counts, uint32_t num_blocks, uint32_t* occupied)
{
    __shared__ uint32_t s[kMaxScanElems];
    for (uint32_t i = threadIdx.x; i < num_blocks; i += blockDim.x) { s[i] = block_counts[i]; }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        uint32_t acc = 0;
        for (uint32_t i = 0; i < num_blocks; ++i) { uint32_t v = s[i]; s[i] = acc; acc += v; }
        *occupied = acc;  // grand total = number of occupied cells
    }
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < num_blocks; i += blockDim.x) { block_counts[i] = s[i]; }
}

// Pass 3 -- write each emitter at its block's base offset + (already written by this block) + block rank.
__global__ void emitWriteKernel(const uint32_t* counts, const unsigned long long* sums, uint64_t nCells,
                                uint32_t N, bool centroid, float* reduced_pos, const uint32_t* block_base)
{
    __shared__ uint32_t s_scan[kWarp + 1];
    __shared__ uint32_t s_written;  // emitters this block has already placed (across iterations)

    const uint32_t base = block_base[blockIdx.x];
    if (threadIdx.x == 0) { s_written = 0u; }
    __syncthreads();

    const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
    const uint64_t start  = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t iters_end = ((nCells + stride - 1) / stride) * stride;
    for (uint64_t c = start; c < iters_end; c += stride)
    {
        const bool emit = (c < nCells) && (counts[c] != 0u);
        uint32_t total = 0;
        const uint32_t rank = blockExclusiveScan(emit, s_scan, total);
        if (emit)
        {
            uint32_t slot = base + s_written + rank;
            // 64-bit base: slot approaches N^3 (< 2^32); slot*3 in 32 bits wraps past ~1.43 B cells and
            // scatters representatives to garbage low offsets (mirrors the AABB-path 64-bit fix).
            size_t off = static_cast<size_t>(slot) * 3;
            float3 rep = cellRepresentative(c, N, centroid, counts, sums);
            reduced_pos[off + 0] = rep.x;
            reduced_pos[off + 1] = rep.y;
            reduced_pos[off + 2] = rep.z;
        }
        __syncthreads();
        if (threadIdx.x == 0) { s_written += total; }
        __syncthreads();
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
    uint32_t* block_counts = nullptr;       // uint32_t[emit_blocks]: per-block emitter counts -> offsets
    int sm_count = 1;                        // SMs on the device; sizes the emit's persistent grid
    int emit_blocks_per_sm = 1;              // max emit blocks resident per SM (occupancy-limited)
    uint32_t emit_blocks = 1;                // persistent-grid block count for the emit passes

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
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
        if (sm_count < 1) { sm_count = 1; }
        // Size the emit's persistent grid to exactly what an SM can hold resident: fewer blocks means
        // fewer block-count entries to scan and, historically, fewer global atomics (now none). Capped
        // at one-thread-per-cell (small grids) and at kMaxScanElems (the single-block scan's capacity).
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&emit_blocks_per_sm, emitWriteKernel, kThreads, 0);
        if (emit_blocks_per_sm < 1) { emit_blocks_per_sm = 1; }
        const uint64_t one_per_cell = (nCells + kThreads - 1) / kThreads;
        uint64_t blocks = static_cast<uint64_t>(sm_count) * emit_blocks_per_sm;
        if (blocks > one_per_cell)  { blocks = one_per_cell; }
        if (blocks > kMaxScanElems) { blocks = kMaxScanElems; }
        if (blocks < 1)             { blocks = 1; }
        emit_blocks = static_cast<uint32_t>(blocks);
        checkCuda(cudaMalloc(&block_counts, emit_blocks * sizeof(uint32_t)), "cudaMalloc block_counts");
    }

    ~Impl()
    {
        cudaFree(counts);
        cudaFree(sums);
        cudaFree(block_counts);
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

        // Atomic-free compaction in three passes on the persistent grid (emit_blocks fixed at ctor time
        // to the SM-resident count, capped at kMaxScanElems): count per block -> scan the block counts
        // into base offsets -> write. No kernel touches a shared global counter (see the emit kernels).
        emitCountKernel<<<emit_blocks, kThreads, 0, stream>>>(counts, nCells, block_counts);
        checkLaunch("emitCountKernel");
        scanBlockCountsKernel<<<1, kThreads, 0, stream>>>(block_counts, emit_blocks, occupied_dev);
        checkLaunch("scanBlockCountsKernel");
        emitWriteKernel<<<emit_blocks, kThreads, 0, stream>>>(
            counts, sums, nCells, gridN, centroid, reduced_pos, block_counts);
        checkLaunch("emitWriteKernel");

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
