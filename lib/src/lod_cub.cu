// LodCub implementation: CUB radix-sort + run-length-encode/reduce-by-key pipeline. See
// lod_cub.hpp for the public contract and .superpowers/sdd/cub-t1-brief.md for the design.
//
// Pipeline (all async on the caller's stream; buffers pre-allocated in the ctor):
//   1. keyKernel:      position -> (cell key, particle index).
//   2. DeviceRadixSort::SortPairs on (key, index) -- groups particles by cell.
//   3. DeviceRunLengthEncode::Encode on the sorted keys -- occupied cell set (uniq), per-cell
//      counts, and the occupied-cell count (written directly into the caller's occupied_dev, no
//      extra device-to-device copy).
//   4. centroid only: gatherKernel builds a sorted-order position array from the sorted indices,
//      then DeviceReduceByKey sums it per unique cell (aligned with `uniq`).
//   5. finalizeKernel: one thread per (bounded) occupied cell -> the representative position
//      (mass centroid or cell geometric center), depending on `centroid`.
//
// Sort-based (not atomic-scatter): the occupied-cell SET and centroids are therefore
// deterministic run-to-run regardless of input particle order.

#include "mimir/lod_cub.hpp"

#include <cub/cub.cuh>

#include <cuda_runtime.h>

#include <algorithm>
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
        std::fprintf(stderr, "LodCub CUDA error in %s: %s\n", what, cudaGetErrorString(err));
        std::abort();
    }
}

// Number of cells (N^3) as a 64-bit value to avoid any 32-bit overflow while computing it (the
// caller-enforced cap gridN <= 1625 keeps the RESULT itself just under 2^32).
uint64_t numCells(uint32_t gridN)
{
    return static_cast<uint64_t>(gridN) * gridN * gridN;
}

// Smallest end_bit such that all keys in [0, numCells(gridN)) are representable, i.e.
// ceil(log2(numCells(gridN))).
int bitsForN(uint32_t gridN)
{
    uint64_t cells = numCells(gridN);
    int bits = 0;
    while ((uint64_t{1} << bits) < cells) ++bits;
    return bits;
}

// float3 addition functor for DeviceReduce::ReduceByKey (per-cell position sums).
struct Float3Add
{
    __host__ __device__ __forceinline__ float3 operator()(const float3& a, const float3& b) const
    {
        return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
    }
};

// Bins each particle into its cell: key = linear cell id, value = particle index (IndexT widens
// to uint64 when max_particles > 2^32; see LodCub ctor). Matches pathtrace_lod_scatter.slang's
// domain mapping exactly: clamp(int((p+1)*0.5*N), 0, N-1) per axis, linear id = cx+N*(cy+N*cz).
template <typename IndexT>
__global__ void keyKernel(const float* pos, uint64_t n, uint32_t N, uint32_t* keys, IndexT* idx)
{
    uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float px = pos[i * 3 + 0];
    float py = pos[i * 3 + 1];
    float pz = pos[i * 3 + 2];
    float fn = static_cast<float>(N);

    auto axisCell = [fn, N](float p) -> uint32_t {
        float v = (p + 1.0f) * 0.5f * fn;
        v = fminf(fmaxf(v, 0.0f), fn - 1.0f);
        return static_cast<uint32_t>(static_cast<int>(v));
    };
    uint32_t cx = axisCell(px);
    uint32_t cy = axisCell(py);
    uint32_t cz = axisCell(pz);

    keys[i] = cx + N * (cy + N * cz);
    idx[i]  = static_cast<IndexT>(i);
}

// Rebuilds a position array in SORTED order (indexed by the radix sort's output permutation) so
// DeviceReduce::ReduceByKey can sum contiguous runs directly.
template <typename IndexT>
__global__ void gatherKernel(const IndexT* idx_sorted, const float* pos, uint64_t n, float3* out)
{
    uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint64_t src = static_cast<uint64_t>(idx_sorted[i]);
    out[i] = make_float3(pos[src * 3 + 0], pos[src * 3 + 1], pos[src * 3 + 2]);
}

// One thread per occupied unique cell -> its representative position. Bounded by *occupied_dev
// (a device pointer, read on-device) rather than a host-known count, so the whole reduce() stays
// async with no host sync between Encode and this kernel; threads beyond the true occupied count
// (up to the host-computed launch upper bound) simply return.
__global__ void finalizeKernel(const uint32_t* uniq, const uint32_t* counts, const float3* sums,
                                const uint32_t* occupied_dev, uint32_t N, bool centroid,
                                float* reduced_pos)
{
    uint32_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= *occupied_dev) return;

    uint32_t lin = uniq[k];
    float3 center;
    if (centroid)
    {
        float invCount = 1.0f / static_cast<float>(counts[k]);
        center = make_float3(sums[k].x * invCount, sums[k].y * invCount, sums[k].z * invCount);
    }
    else
    {
        uint32_t cx = lin % N;
        uint32_t cy = (lin / N) % N;
        uint32_t cz = lin / (N * N);
        float cs = 2.0f / static_cast<float>(N);
        center = make_float3(-1.0f + (static_cast<float>(cx) + 0.5f) * cs,
                              -1.0f + (static_cast<float>(cy) + 0.5f) * cs,
                              -1.0f + (static_cast<float>(cz) + 0.5f) * cs);
    }
    reduced_pos[k * 3 + 0] = center.x;
    reduced_pos[k * 3 + 1] = center.y;
    reduced_pos[k * 3 + 2] = center.z;
}

constexpr int kThreads = 256;

inline uint32_t gridFor(uint64_t n, int threads = kThreads)
{
    return static_cast<uint32_t>((n + static_cast<uint64_t>(threads) - 1) / threads);
}

// cub::DeviceRunLengthEncode::Encode and cub::DeviceReduce::ReduceByKey (this CUB version) take a
// plain `int`/(offset_t-castable) item count, so the particle count they see must fit in int32.
// This bounds `max_particles` for THIS module independent of the uint32/uint64 index widening
// (which only concerns the value channel of the radix sort). A future task would need chunked
// Encode/ReduceByKey passes to lift this; out of scope here (Task 1's test is 200k particles).
int cubItemCount(uint64_t n)
{
    if (n > static_cast<uint64_t>(INT32_MAX))
    {
        std::fprintf(stderr, "LodCub: particle count %llu exceeds INT32_MAX; unsupported by this "
                     "CUB version's Encode/ReduceByKey item-count parameter\n",
                     static_cast<unsigned long long>(n));
        std::abort();
    }
    return static_cast<int>(n);
}

// Every buffer LodCub::Impl allocates, sized once from (max_particles, gridN, centroid). Shared by
// the ctor (which actually allocates) and scratchBytes() (which only sums sizes) so the two stay
// in lockstep by construction.
struct Sizes
{
    size_t keys = 0;       // uint32_t[max_particles]
    size_t keys_out = 0;   // uint32_t[max_particles]
    size_t idx = 0;        // IndexT[max_particles]
    size_t idx_out = 0;    // IndexT[max_particles]
    size_t uniq = 0;       // uint32_t[maxUnique]
    size_t counts = 0;     // uint32_t[maxUnique]
    size_t sums = 0;       // float3[maxUnique]           (centroid only)
    size_t gathered = 0;   // float3[max_particles]       (centroid only)
    size_t num_runs2 = 0;  // uint32_t (ReduceByKey's own run counter; centroid only)
    size_t cub_temp = 0;   // shared scratch for SortPairs / Encode / ReduceByKey

    size_t total() const
    {
        return keys + keys_out + idx + idx_out + uniq + counts + sums + gathered + num_runs2
             + cub_temp;
    }
};

template <typename IndexT>
Sizes computeSizes(uint64_t max_particles, uint32_t gridN, bool centroid)
{
    Sizes s;
    uint64_t maxUnique = std::min(numCells(gridN), max_particles);
    int bits = bitsForN(gridN);
    int items = cubItemCount(max_particles);

    s.keys     = max_particles * sizeof(uint32_t);
    s.keys_out = max_particles * sizeof(uint32_t);
    s.idx      = max_particles * sizeof(IndexT);
    s.idx_out  = max_particles * sizeof(IndexT);
    s.uniq     = maxUnique * sizeof(uint32_t);
    s.counts   = maxUnique * sizeof(uint32_t);

    size_t sort_temp = 0;
    checkCuda(cub::DeviceRadixSort::SortPairs<uint32_t, IndexT>(
        nullptr, sort_temp,
        static_cast<const uint32_t*>(nullptr), static_cast<uint32_t*>(nullptr),
        static_cast<const IndexT*>(nullptr), static_cast<IndexT*>(nullptr),
        max_particles, 0, bits), "SortPairs size query");

    size_t encode_temp = 0;
    checkCuda(cub::DeviceRunLengthEncode::Encode(
        nullptr, encode_temp,
        static_cast<const uint32_t*>(nullptr), static_cast<uint32_t*>(nullptr),
        static_cast<uint32_t*>(nullptr), static_cast<uint32_t*>(nullptr),
        items), "Encode size query");

    size_t reduce_temp = 0;
    if (centroid)
    {
        s.sums     = maxUnique * sizeof(float3);
        s.gathered = max_particles * sizeof(float3);
        s.num_runs2 = sizeof(uint32_t);

        checkCuda(cub::DeviceReduce::ReduceByKey(
            nullptr, reduce_temp,
            static_cast<const uint32_t*>(nullptr), static_cast<uint32_t*>(nullptr),
            static_cast<const float3*>(nullptr), static_cast<float3*>(nullptr),
            static_cast<uint32_t*>(nullptr), Float3Add{}, items),
            "ReduceByKey size query");
    }

    s.cub_temp = std::max({sort_temp, encode_temp, reduce_temp});
    return s;
}

Sizes computeSizesDispatch(uint64_t max_particles, uint32_t gridN, bool centroid)
{
    if (max_particles > static_cast<uint64_t>(UINT32_MAX))
    {
        return computeSizes<uint64_t>(max_particles, gridN, centroid);
    }
    return computeSizes<uint32_t>(max_particles, gridN, centroid);
}

} // namespace

struct LodCub::Impl
{
    uint64_t max_particles;
    uint32_t gridN;
    bool centroid;
    bool wide_index; // true -> per-particle index (sort value) is uint64, else uint32
    uint64_t max_unique;
    int bits;

    void* keys = nullptr;      // uint32_t[max_particles]
    void* keys_out = nullptr;  // uint32_t[max_particles]
    void* idx = nullptr;       // IndexT[max_particles]
    void* idx_out = nullptr;   // IndexT[max_particles]
    void* uniq = nullptr;      // uint32_t[max_unique]
    void* counts = nullptr;    // uint32_t[max_unique]
    void* sums = nullptr;      // float3[max_unique]      (centroid only)
    void* gathered = nullptr;  // float3[max_particles]   (centroid only)
    void* num_runs2 = nullptr; // uint32_t                (centroid only, ReduceByKey's run count)
    void* cub_temp = nullptr;
    size_t cub_temp_bytes = 0;

    Impl(uint64_t max_particles_, uint32_t gridN_, bool centroid_)
        : max_particles(max_particles_), gridN(gridN_), centroid(centroid_),
          wide_index(max_particles_ > static_cast<uint64_t>(UINT32_MAX)),
          max_unique(std::min(numCells(gridN_), max_particles_)),
          bits(bitsForN(gridN_))
    {
        Sizes s = computeSizesDispatch(max_particles, gridN, centroid);

        checkCuda(cudaMalloc(&keys, s.keys), "cudaMalloc keys");
        checkCuda(cudaMalloc(&keys_out, s.keys_out), "cudaMalloc keys_out");
        checkCuda(cudaMalloc(&idx, s.idx), "cudaMalloc idx");
        checkCuda(cudaMalloc(&idx_out, s.idx_out), "cudaMalloc idx_out");
        checkCuda(cudaMalloc(&uniq, s.uniq), "cudaMalloc uniq");
        checkCuda(cudaMalloc(&counts, s.counts), "cudaMalloc counts");
        if (centroid)
        {
            checkCuda(cudaMalloc(&sums, s.sums), "cudaMalloc sums");
            checkCuda(cudaMalloc(&gathered, s.gathered), "cudaMalloc gathered");
            checkCuda(cudaMalloc(&num_runs2, s.num_runs2), "cudaMalloc num_runs2");
        }
        cub_temp_bytes = s.cub_temp;
        if (cub_temp_bytes > 0)
        {
            checkCuda(cudaMalloc(&cub_temp, cub_temp_bytes), "cudaMalloc cub_temp");
        }
    }

    ~Impl()
    {
        cudaFree(keys);
        cudaFree(keys_out);
        cudaFree(idx);
        cudaFree(idx_out);
        cudaFree(uniq);
        cudaFree(counts);
        cudaFree(sums);
        cudaFree(gathered);
        cudaFree(num_runs2);
        cudaFree(cub_temp);
    }

    template <typename IndexT>
    void reduceTyped(cudaStream_t stream, const float* positions, uint64_t count,
                      float* reduced_pos, uint32_t* occupied_dev)
    {
        auto* keysT = static_cast<uint32_t*>(keys);
        auto* keysOutT = static_cast<uint32_t*>(keys_out);
        auto* idxT = static_cast<IndexT*>(idx);
        auto* idxOutT = static_cast<IndexT*>(idx_out);
        auto* uniqT = static_cast<uint32_t*>(uniq);
        auto* countsT = static_cast<uint32_t*>(counts);

        // 1. key each particle: (cell id, particle index).
        keyKernel<IndexT><<<gridFor(count), kThreads, 0, stream>>>(
            positions, count, gridN, keysT, idxT);

        // 2. sort by cell id (order-independent grouping -> deterministic occupied set).
        size_t sort_temp_bytes = cub_temp_bytes;
        checkCuda(cub::DeviceRadixSort::SortPairs<uint32_t, IndexT>(
            cub_temp, sort_temp_bytes, keysT, keysOutT, idxT, idxOutT, count, 0, bits, stream),
            "SortPairs");

        // 3. run-length-encode the sorted keys -> occupied set + counts + occupied count (written
        //    directly into the caller's occupied_dev; no extra device-to-device copy needed).
        size_t encode_temp_bytes = cub_temp_bytes;
        checkCuda(cub::DeviceRunLengthEncode::Encode(
            cub_temp, encode_temp_bytes, keysOutT, uniqT, countsT, occupied_dev,
            cubItemCount(count), stream), "Encode");

        // 4. centroid only: gather positions into sorted order, then sum per unique cell.
        if (centroid)
        {
            auto* gatheredT = static_cast<float3*>(gathered);
            auto* sumsT = static_cast<float3*>(sums);
            auto* numRuns2T = static_cast<uint32_t*>(num_runs2);

            gatherKernel<IndexT><<<gridFor(count), kThreads, 0, stream>>>(
                idxOutT, positions, count, gatheredT);

            size_t reduce_temp_bytes = cub_temp_bytes;
            // Reuses `uniqT` as the (redundant but harmless) unique-key output: ReduceByKey walks
            // the same sorted key sequence Encode just consumed, so it re-derives identical keys.
            checkCuda(cub::DeviceReduce::ReduceByKey(
                cub_temp, reduce_temp_bytes, keysOutT, uniqT, gatheredT, sumsT, numRuns2T,
                Float3Add{}, cubItemCount(count), stream), "ReduceByKey");
        }

        // 5. finalize: one thread per occupied cell (bounded by min(N^3, count), read on-device
        //    via occupied_dev so no host sync is needed between steps 3/4 and this launch).
        uint64_t upperBound = std::min(numCells(gridN), count);
        finalizeKernel<<<gridFor(upperBound), kThreads, 0, stream>>>(
            uniqT, countsT, static_cast<const float3*>(sums), occupied_dev, gridN, centroid,
            reduced_pos);
    }

    void reduce(cudaStream_t stream, const void* positions_dev, uint64_t count,
                void* reduced_pos_dev, uint32_t* occupied_dev)
    {
        const float* positions = static_cast<const float*>(positions_dev);
        float* reduced_pos = static_cast<float*>(reduced_pos_dev);
        if (wide_index)
        {
            reduceTyped<uint64_t>(stream, positions, count, reduced_pos, occupied_dev);
        }
        else
        {
            reduceTyped<uint32_t>(stream, positions, count, reduced_pos, occupied_dev);
        }
    }
};

LodCub::LodCub(uint64_t max_particles, uint32_t gridN, bool centroid)
    : impl(new Impl(max_particles, gridN, centroid))
{
}

LodCub::~LodCub()
{
    delete impl;
}

size_t LodCub::scratchBytes(uint64_t max_particles, uint32_t gridN, bool centroid)
{
    return computeSizesDispatch(max_particles, gridN, centroid).total();
}

void LodCub::reduce(cudaStream_t stream, const void* positions_dev, uint64_t count,
                     void* reduced_pos_dev, uint32_t* occupied_dev)
{
    impl->reduce(stream, positions_dev, count, reduced_pos_dev, occupied_dev);
}

} // namespace mimir
