#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

// Random-walk 3D point cloud, shared by benchmark_mimir and benchmark_datoviz.
// Points live in the [-1,1]^3 cube (centered at the origin so both renderers show
// the same centered cloud).
//
// Initial positions follow a k-modal gaussian mixture: k cluster centers drawn
// uniformly over the whole cube (deterministic from the seed), each point assigned
// to a random cluster and offset by a per-axis gaussian of stddev epsilon. Small
// epsilon + moderate k gives a blobby, cheese-like geometry: cheap to simulate
// but demanding to render (dense occluded clusters — the path-tracing target).
// Blobs whose center lands near a wall are sliced flat by the [-1,1] clamp, so
// the bounding cube itself reads in the render.
//
// The per-step walk is mean-reverting (Ornstein-Uhlenbeck): each point adds a
// gaussian displacement AND is pulled back toward its cluster center, with the
// reversion rate chosen so the stationary per-axis stddev stays epsilon. A pure
// Brownian walk would diffuse the clusters into a uniform cloud within ~1k steps;
// this keeps the cheese shape indefinitely while the points still jiggle.

struct PointsParams {
    unsigned int count   = 1'000'000;
    uint32_t     seed    = 12345;
    unsigned int k       = 8;      // number of gaussian modes (clusters)
    float        epsilon = 0.05f;  // per-axis stddev of each mode, in domain units
};

// Persistent curand states, sized to fill the device (one state per resident thread).
// `states` is a curandState* kept opaque so this header stays includable from .cpp TUs.
struct RngStates {
    void* states = nullptr;
    int   grid   = 0;
    int   block  = 0;
    int   count  = 0;
};

RngStates createRngStates(uint32_t seed);
void destroyRngStates(RngStates& rng);
size_t rngStatesBytes(const RngStates& rng);

// Persistent cluster data for the mean-reverting walk: the k centers (float3*) and
// the per-point cluster assignment (unsigned int*), both device buffers kept opaque.
// lambda is the OU reversion rate derived from epsilon (see points3d_sim.cu).
struct ClusterData {
    void*        centers = nullptr;
    void*        ids     = nullptr;
    unsigned int k       = 1;
    float        lambda  = 0.f;
};

// Centers come from a host-side RNG seeded with params.seed so both benchmarks see
// the same cloud; ids are filled by launchInitPositions.
ClusterData createClusters(const PointsParams& params);
void destroyClusters(ClusterData& clusters);
size_t clusterBytes(const ClusterData& clusters, size_t point_count);

// K-modal initial positions (see header comment) — deterministic from params.seed;
// also records each point's cluster id for the mean-reverting walk.
void launchInitPositions(float* pos3, const PointsParams& params,
                         ClusterData& clusters, RngStates& rng);

// One walk step: per-axis gaussian displacement plus OU pull toward the point's
// cluster center, clamped to the cube.
void launchIntegrate3D(float* pos3, size_t point_count, const ClusterData& clusters,
                       RngStates& rng, cudaStream_t s = 0);
