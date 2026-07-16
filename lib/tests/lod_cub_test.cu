// Parity test for LodCub (Task 1 of the CUB-LOD migration): generates a deterministic host point
// cloud, runs LodCub::reduce on the GPU, and compares the occupied-cell count + representative
// positions against a CPU reference that bins the same points with the identical domain mapping.
// Run for BOTH centroid=true and centroid=false placements. See .superpowers/sdd/cub-t1-brief.md.

#include "mimir/lod_cub.hpp"

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <map>
#include <random>
#include <vector>

namespace
{

void checkCuda(cudaError_t err, const char* what)
{
    if (err != cudaSuccess)
    {
        std::fprintf(stderr, "CUDA error in %s: %s\n", what, cudaGetErrorString(err));
        std::exit(1);
    }
}

// Deterministic point cloud: a handful of gaussian clusters inside [-1,1]^3 (clamped so every
// point stays in-domain), generated with a fixed seed so the test is fully reproducible.
std::vector<float> buildPoints(uint64_t n)
{
    std::vector<float> pts(n * 3);
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> centerDist(-0.6f, 0.6f);
    std::normal_distribution<float> spread(0.0f, 0.12f);

    constexpr int kClusters = 6;
    std::vector<std::array<float, 3>> centers;
    for (int c = 0; c < kClusters; ++c)
    {
        centers.push_back({centerDist(rng), centerDist(rng), centerDist(rng)});
    }

    for (uint64_t i = 0; i < n; ++i)
    {
        const auto& c = centers[i % kClusters];
        for (int axis = 0; axis < 3; ++axis)
        {
            float v = c[static_cast<size_t>(axis)] + spread(rng);
            v = std::fmin(std::fmax(v, -1.0f), 1.0f);
            pts[i * 3 + static_cast<uint64_t>(axis)] = v;
        }
    }
    return pts;
}

// Cell index per axis: clamp(int((p + 1.0) * 0.5 * N), 0, N-1); linear id = cx + N*(cy + N*cz).
// Matches pathtrace_lod_scatter.slang exactly.
uint32_t cellIndex(float p, uint32_t N)
{
    float fn = static_cast<float>(N);
    int c = static_cast<int>((p + 1.0f) * 0.5f * fn);
    if (c < 0) c = 0;
    if (c > static_cast<int>(N) - 1) c = static_cast<int>(N) - 1;
    return static_cast<uint32_t>(c);
}

uint32_t linearCell(const float* p, uint32_t N)
{
    uint32_t cx = cellIndex(p[0], N);
    uint32_t cy = cellIndex(p[1], N);
    uint32_t cz = cellIndex(p[2], N);
    return cx + N * (cy + N * cz);
}

struct CpuCell
{
    uint32_t count = 0;
    double sum[3] = {0.0, 0.0, 0.0};
};

struct CpuRef
{
    uint32_t occupied = 0;
    // key (linear cell id) -> representative position (centroid or cell center).
    std::map<uint32_t, std::array<float, 3>> reps;
};

CpuRef cpuReduce(const std::vector<float>& pts, uint32_t N, bool centroid)
{
    std::map<uint32_t, CpuCell> cells;
    uint64_t n = pts.size() / 3;
    for (uint64_t i = 0; i < n; ++i)
    {
        const float* p = &pts[i * 3];
        uint32_t lin = linearCell(p, N);
        CpuCell& cell = cells[lin];
        cell.count++;
        cell.sum[0] += p[0];
        cell.sum[1] += p[1];
        cell.sum[2] += p[2];
    }

    CpuRef ref;
    ref.occupied = static_cast<uint32_t>(cells.size());
    float cs = 2.0f / static_cast<float>(N);
    for (const auto& [lin, cell] : cells)
    {
        std::array<float, 3> rep{};
        if (centroid)
        {
            rep[0] = static_cast<float>(cell.sum[0] / cell.count);
            rep[1] = static_cast<float>(cell.sum[1] / cell.count);
            rep[2] = static_cast<float>(cell.sum[2] / cell.count);
        }
        else
        {
            uint32_t cx = lin % N;
            uint32_t cy = (lin / N) % N;
            uint32_t cz = lin / (N * N);
            rep[0] = -1.0f + (static_cast<float>(cx) + 0.5f) * cs;
            rep[1] = -1.0f + (static_cast<float>(cy) + 0.5f) * cs;
            rep[2] = -1.0f + (static_cast<float>(cz) + 0.5f) * cs;
        }
        ref.reps[lin] = rep;
    }
    return ref;
}

// Run LodCub for one placement and compare against the CPU reference. Returns true on success.
bool runCase(const std::vector<float>& pts, uint32_t gridN, bool centroid)
{
    uint64_t n = pts.size() / 3;
    CpuRef ref = cpuReduce(pts, gridN, centroid);

    float* positions_dev = nullptr;
    float* reduced_pos_dev = nullptr;
    uint32_t* occupied_dev = nullptr;
    checkCuda(cudaMalloc(&positions_dev, pts.size() * sizeof(float)), "cudaMalloc positions");
    // Upper bound on occupied cells is min(N^3, n); allocate generously for n.
    checkCuda(cudaMalloc(&reduced_pos_dev, n * 3 * sizeof(float)), "cudaMalloc reduced_pos");
    checkCuda(cudaMalloc(&occupied_dev, sizeof(uint32_t)), "cudaMalloc occupied");
    checkCuda(cudaMemcpy(positions_dev, pts.data(), pts.size() * sizeof(float),
                          cudaMemcpyHostToDevice), "memcpy positions");

    cudaStream_t stream = nullptr;
    checkCuda(cudaStreamCreate(&stream), "stream create");

    mimir::LodCub lod(n, gridN, centroid);
    lod.reduce(stream, positions_dev, n, reduced_pos_dev, occupied_dev);
    checkCuda(cudaStreamSynchronize(stream), "stream sync");

    uint32_t occupied = 0;
    checkCuda(cudaMemcpy(&occupied, occupied_dev, sizeof(uint32_t), cudaMemcpyDeviceToHost),
              "memcpy occupied");
    std::vector<float> reduced(static_cast<size_t>(occupied) * 3);
    if (occupied > 0)
    {
        checkCuda(cudaMemcpy(reduced.data(), reduced_pos_dev, reduced.size() * sizeof(float),
                              cudaMemcpyDeviceToHost), "memcpy reduced");
    }

    checkCuda(cudaStreamDestroy(stream), "stream destroy");
    cudaFree(positions_dev);
    cudaFree(reduced_pos_dev);
    cudaFree(occupied_dev);

    const char* label = centroid ? "centroid=true" : "centroid=false";

    if (occupied != ref.occupied)
    {
        std::fprintf(stderr, "[%s] FAIL: occupied count mismatch: gpu=%u cpu=%u\n",
                     label, occupied, ref.occupied);
        return false;
    }

    float cellSize = 2.0f / static_cast<float>(gridN);
    double tol = 2.0 * cellSize / 1073741824.0 /* 2^30 */ + 1e-6;

    // Every GPU representative must land in an occupied cell of the CPU reference and match its
    // centroid/center within tolerance. Match by the cell the GPU representative itself falls into
    // (recomputing its cell id), since GPU/CPU emit order need not match.
    std::map<uint32_t, bool> seen;
    for (uint32_t k = 0; k < occupied; ++k)
    {
        const float* rp = &reduced[static_cast<size_t>(k) * 3];
        uint32_t lin = linearCell(rp, gridN);
        auto it = ref.reps.find(lin);
        if (it == ref.reps.end())
        {
            std::fprintf(stderr, "[%s] FAIL: gpu rep %u at cell %u is not an occupied cell in the "
                         "CPU reference\n", label, k, lin);
            return false;
        }
        if (seen.count(lin))
        {
            std::fprintf(stderr, "[%s] FAIL: cell %u produced by GPU more than once\n", label, lin);
            return false;
        }
        seen[lin] = true;
        const auto& cpuRep = it->second;
        for (int axis = 0; axis < 3; ++axis)
        {
            double diff = std::fabs(static_cast<double>(rp[axis]) - static_cast<double>(cpuRep[static_cast<size_t>(axis)]));
            if (diff > tol)
            {
                std::fprintf(stderr, "[%s] FAIL: cell %u axis %d mismatch: gpu=%f cpu=%f diff=%g "
                             "tol=%g\n", label, lin, axis, rp[axis], cpuRep[static_cast<size_t>(axis)], diff, tol);
                return false;
            }
        }
    }
    if (seen.size() != ref.reps.size())
    {
        std::fprintf(stderr, "[%s] FAIL: gpu emitted %zu distinct cells, cpu reference has %zu\n",
                     label, seen.size(), ref.reps.size());
        return false;
    }

    std::printf("[%s] PASS: occupied=%u matches CPU reference (tol=%g)\n", label, occupied, tol);
    return true;
}

} // namespace

int main()
{
    constexpr uint64_t kNumPoints = 200000;
    constexpr uint32_t kGridN = 16;

    std::vector<float> pts = buildPoints(kNumPoints);

    bool ok = true;
    ok = runCase(pts, kGridN, /*centroid=*/true) && ok;
    ok = runCase(pts, kGridN, /*centroid=*/false) && ok;

    if (!ok)
    {
        std::fprintf(stderr, "lod_cub_test: FAILED\n");
        return 1;
    }
    std::printf("lod_cub_test: ALL PASS\n");
    return 0;
}
