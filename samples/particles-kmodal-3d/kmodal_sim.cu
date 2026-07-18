#include "kmodal_sim.cuh"
#include <curand_kernel.h>
#include <random>
#include <vector>
#include <cstdlib> // getenv (MIMIR_SIM_SORT_KTIME)
#include <cstdio>  // fprintf (sort stage timing)

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

// Displacement stddev per axis per step, in world units (cube spans [-1,1]). Small = gentle,
// slow per-iteration motion; the reversion rate lambda is derived from this, so the equilibrium
// blob size (epsilon) is unaffected -- the clusters just evolve more slowly.
constexpr float STEP_SIGMA = 0.001f;

__global__ void initRngKernel(curandState* states, int state_count, uint32_t seed)
{
    int tidx = (int)(blockDim.x * blockIdx.x + threadIdx.x);
    if (tidx >= state_count) return;
    curand_init(seed, tidx, 0, &states[tidx]);
}

__global__ void initPosKernel(float* coords, size_t point_count, const float3* centers,
                              unsigned int* ids, unsigned int k, float epsilon,
                              curandState* rng)
{
    auto points = reinterpret_cast<float3*>(coords);
    auto tidx   = blockDim.x * blockIdx.x + threadIdx.x;
    auto stride = gridDim.x * blockDim.x;
    auto state  = rng[tidx];
    for (size_t i = tidx; i < point_count; i += stride)
    {
        // curand_uniform is in (0,1], so u*k can reach k exactly; clamp to k-1.
        auto c = min((unsigned int)(curand_uniform(&state) * k), k - 1);
        ids[i] = c;
        auto ctr = centers[c];
        points[i] = {
            fminf(fmaxf(ctr.x + epsilon * curand_normal(&state), -1.f), 1.f),
            fminf(fmaxf(ctr.y + epsilon * curand_normal(&state), -1.f), 1.f),
            fminf(fmaxf(ctr.z + epsilon * curand_normal(&state), -1.f), 1.f),
        };
    }
    rng[tidx] = state;
}

__global__ void integrate3dKernel(float* coords, size_t point_count,
                                  const float3* centers, const unsigned int* ids,
                                  float lambda, curandState* rng)
{
    auto points = reinterpret_cast<float3*>(coords);
    auto tidx   = blockDim.x * blockIdx.x + threadIdx.x;
    auto stride = gridDim.x * blockDim.x;
    auto state  = rng[tidx];
    for (size_t i = tidx; i < point_count; i += stride)
    {
        auto p   = points[i];
        auto ctr = centers[ids[i]];
        p.x += STEP_SIGMA * curand_normal(&state) - lambda * (p.x - ctr.x);
        p.y += STEP_SIGMA * curand_normal(&state) - lambda * (p.y - ctr.y);
        p.z += STEP_SIGMA * curand_normal(&state) - lambda * (p.z - ctr.z);
        p.x = fminf(fmaxf(p.x, -1.f), 1.f);
        p.y = fminf(fmaxf(p.y, -1.f), 1.f);
        p.z = fminf(fmaxf(p.z, -1.f), 1.f);
        points[i] = p;
    }
    rng[tidx] = state;
}

// ---------------------------------------------------------------------------
// Experimental spatial sort (see kmodal_sim.cuh)
// ---------------------------------------------------------------------------
static constexpr int kSortThreads = 256;
static inline uint32_t sortGridFor(uint64_t n){ uint64_t b=(n+kSortThreads-1)/kSortThreads; if(b>2147483647ull)b=2147483647ull; if(b<1)b=1; return (uint32_t)b; }

__device__ __forceinline__ uint32_t sortCellId(float px, float py, float pz, uint32_t N)
{
    float n=(float)N;
    int cx=(int)fminf(fmaxf((px+1.f)*0.5f*n,0.f),n-1.f);
    int cy=(int)fminf(fmaxf((py+1.f)*0.5f*n,0.f),n-1.f);
    int cz=(int)fminf(fmaxf((pz+1.f)*0.5f*n,0.f),n-1.f);
    return (uint32_t)cx + N*((uint32_t)cy + N*(uint32_t)cz);
}
__global__ void sortClearKernel(uint32_t* a, uint64_t n){
    for(uint64_t i=(uint64_t)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(uint64_t)blockDim.x*gridDim.x) a[i]=0u; }
__global__ void sortHistKernel(const float* pos, size_t count, uint32_t N, uint32_t* hist){
    for(uint64_t i=(uint64_t)blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=(uint64_t)blockDim.x*gridDim.x)
        atomicAdd(&hist[sortCellId(pos[i*3],pos[i*3+1],pos[i*3+2],N)],1u); }
// exclusive scan pieces (blockSums must be scanned by sortScanSingle, then added back)
__global__ void sortScanLocalKernel(const uint32_t* in, uint32_t* out, uint32_t* blockSums, uint64_t n){
    __shared__ uint32_t s[kSortThreads];
    uint64_t g=(uint64_t)blockIdx.x*blockDim.x+threadIdx.x; uint32_t v=(g<n)?in[g]:0u; s[threadIdx.x]=v; __syncthreads();
    for(uint32_t o=1;o<blockDim.x;o<<=1){ uint32_t t=(threadIdx.x>=o)?s[threadIdx.x-o]:0u; __syncthreads(); s[threadIdx.x]+=t; __syncthreads(); }
    if(g<n) out[g]=s[threadIdx.x]-v; if(threadIdx.x==blockDim.x-1) blockSums[blockIdx.x]=s[threadIdx.x]; }
__global__ void sortScanSingleKernel(uint32_t* data, uint32_t L){
    __shared__ uint32_t s[kSortThreads]; __shared__ uint32_t run; if(threadIdx.x==0) run=0u; __syncthreads();
    for(uint32_t base=0;base<L;base+=blockDim.x){ uint32_t idx=base+threadIdx.x; uint32_t v=(idx<L)?data[idx]:0u; s[threadIdx.x]=v; __syncthreads();
        for(uint32_t o=1;o<blockDim.x;o<<=1){ uint32_t t=(threadIdx.x>=o)?s[threadIdx.x-o]:0u; __syncthreads(); s[threadIdx.x]+=t; __syncthreads(); }
        if(idx<L) data[idx]=s[threadIdx.x]-v+run; __syncthreads(); if(threadIdx.x==0) run+=s[blockDim.x-1]; __syncthreads(); } }
__global__ void sortAddOffsetsKernel(uint32_t* out, const uint32_t* blockScan, uint64_t n){
    uint64_t g=(uint64_t)blockIdx.x*blockDim.x+threadIdx.x; if(g<n) out[g]+=blockScan[blockIdx.x]; }
__global__ void sortPlaceKernel(const float* pos, const unsigned int* ids, size_t count, uint32_t N,
                                uint32_t* cursor, float* posOut, unsigned int* idsOut){
    for(uint64_t i=(uint64_t)blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=(uint64_t)blockDim.x*gridDim.x){
        float px=pos[i*3],py=pos[i*3+1],pz=pos[i*3+2]; uint32_t c=sortCellId(px,py,pz,N);
        uint32_t d=atomicAdd(&cursor[c],1u); posOut[3*(uint64_t)d]=px; posOut[3*(uint64_t)d+1]=py; posOut[3*(uint64_t)d+2]=pz; idsOut[d]=ids[i]; } }

SortScratch createSortScratch(size_t count, uint32_t sortN)
{
    SortScratch s; s.sortN=sortN; s.count=count; s.nCells=(uint64_t)sortN*sortN*sortN;
    uint32_t scanBlocks=sortGridFor(s.nCells);
    cudaMalloc(&s.hist,       s.nCells*sizeof(uint32_t));
    cudaMalloc(&s.off,        s.nCells*sizeof(uint32_t));
    cudaMalloc(&s.blockSums,  (size_t)scanBlocks*sizeof(uint32_t));
    cudaMalloc(&s.cursor,     s.nCells*sizeof(uint32_t));
    cudaMalloc(&s.pos_sorted, count*3*sizeof(float));
    cudaMalloc(&s.ids_sorted, count*sizeof(unsigned int));
    return s;
}
void destroySortScratch(SortScratch& s)
{
    cudaFree(s.hist); cudaFree(s.off); cudaFree(s.blockSums); cudaFree(s.cursor);
    cudaFree(s.pos_sorted); cudaFree(s.ids_sorted);
    s = SortScratch{};
}
size_t sortScratchBytes(const SortScratch& s)
{
    uint32_t scanBlocks=sortGridFor(s.nCells);
    return (3*s.nCells + scanBlocks)*sizeof(uint32_t) + s.count*3*sizeof(float) + s.count*sizeof(unsigned int);
}
void launchSpatialSort(float* pos3, unsigned int* ids, size_t count, SortScratch& s, cudaStream_t stream)
{
    const uint32_t N=s.sortN; const uint64_t nC=s.nCells;
    auto* hist=(uint32_t*)s.hist; auto* off=(uint32_t*)s.off; auto* bsum=(uint32_t*)s.blockSums;
    auto* cursor=(uint32_t*)s.cursor; auto* posS=(float*)s.pos_sorted; auto* idsS=(unsigned int*)s.ids_sorted;
    uint32_t scanBlocks=sortGridFor(nC);
    // Persistent grid for the per-particle passes (hist/place): occupancy-sized, grid-strided -- like
    // the LOD scatter -- instead of 781K one-shot blocks, which schedule poorly on the atomic passes.
    static int sm=0; if(!sm){ int d; cudaGetDevice(&d); cudaDeviceGetAttribute(&sm,cudaDevAttrMultiProcessorCount,d); }
    uint32_t partBlocks=sortGridFor(count); if(partBlocks>(uint32_t)sm*32) partBlocks=(uint32_t)sm*32;
    static const bool ktime = (std::getenv("MIMIR_SIM_SORT_KTIME") != nullptr);
    cudaEvent_t e[6]{}; if(ktime){ for(auto& ev:e) cudaEventCreate(&ev); cudaEventRecord(e[0],stream); }
    sortClearKernel<<<sortGridFor(nC),kSortThreads,0,stream>>>(hist,nC);
    sortHistKernel<<<partBlocks,kSortThreads,0,stream>>>(pos3,count,N,hist);
    if(ktime) cudaEventRecord(e[1],stream);
    sortScanLocalKernel<<<scanBlocks,kSortThreads,0,stream>>>(hist,off,bsum,nC);
    sortScanSingleKernel<<<1,kSortThreads,0,stream>>>(bsum,scanBlocks);
    sortAddOffsetsKernel<<<scanBlocks,kSortThreads,0,stream>>>(off,bsum,nC);
    cudaMemcpyAsync(cursor,off,nC*sizeof(uint32_t),cudaMemcpyDeviceToDevice,stream);
    if(ktime) cudaEventRecord(e[2],stream);
    sortPlaceKernel<<<partBlocks,kSortThreads,0,stream>>>(pos3,ids,count,N,cursor,posS,idsS);
    if(ktime) cudaEventRecord(e[3],stream);
    // copy the sorted shadow back into the live interop position buffer + ids
    cudaMemcpyAsync(pos3,posS,count*3*sizeof(float),cudaMemcpyDeviceToDevice,stream);
    if(ktime) cudaEventRecord(e[4],stream);
    cudaMemcpyAsync(ids,idsS,count*sizeof(unsigned int),cudaMemcpyDeviceToDevice,stream);
    if(ktime){ cudaEventRecord(e[5],stream); cudaEventSynchronize(e[5]);
        float t1,t2,t3,t4,t5; cudaEventElapsedTime(&t1,e[0],e[1]); cudaEventElapsedTime(&t2,e[1],e[2]);
        cudaEventElapsedTime(&t3,e[2],e[3]); cudaEventElapsedTime(&t4,e[3],e[4]); cudaEventElapsedTime(&t5,e[4],e[5]);
        std::fprintf(stderr,"[sim-sort-ktime] clear+hist %.2f | scan %.2f | place %.2f | copyback-pos %.2f | copyback-ids %.2f ms\n",t1,t2,t3,t4,t5);
        for(auto& ev:e) cudaEventDestroy(ev); }
}

// ---------------------------------------------------------------------------
// Host API
// ---------------------------------------------------------------------------

RngStates createRngStates(uint32_t seed)
{
    int device_id = -1;
    cudaGetDevice(&device_id);

    int sm_count = -1, max_sm_threads = -1;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id);
    cudaDeviceGetAttribute(&max_sm_threads, cudaDevAttrMaxThreadsPerMultiProcessor, device_id);

    RngStates rng;
    rng.block = 256;
    // One persistent full-occupancy wave (one RNG state per resident thread); each thread grid-strides
    // over count/threads particles. Measured: oversubscribing the grid (more waves, fewer iterations
    // each) does NOT help the throughput fall-off at huge N -- it was neutral-to-worse -- so the
    // large-N slowdown is a memory-reach (TLB/DRAM) effect, not warp drift, and this stays minimal.
    rng.count = (max_sm_threads / rng.block) * sm_count * rng.block;
    rng.grid  = (rng.count + rng.block - 1) / rng.block;

    curandState* states = nullptr;
    cudaMalloc(&states, sizeof(curandState) * rng.count);
    rng.states = states;

    initRngKernel<<<rng.grid, rng.block>>>(states, rng.count, seed);
    cudaDeviceSynchronize();
    return rng;
}

void destroyRngStates(RngStates& rng)
{
    cudaFree(rng.states);
    rng.states = nullptr;
}

size_t rngStatesBytes(const RngStates& rng)
{
    return sizeof(curandState) * (size_t)rng.count;
}

ClusterData createClusters(const PointsParams& params)
{
    ClusterData clusters;
    clusters.k = params.k > 0 ? params.k : 1;

    // OU reversion rate: the discrete walk p += sigma*N(0,1) - lambda*(p - ctr) has
    // stationary per-axis stddev sigma/sqrt(2*lambda - lambda^2); solving for stddev
    // == epsilon gives lambda ~= sigma^2 / (2*epsilon^2) (small-lambda limit).
    // Capped at 1 (lambda == 1 snaps straight to the center each step).
    if (params.epsilon > 0.f)
        clusters.lambda = fminf(
            (STEP_SIGMA * STEP_SIGMA) / (2.f * params.epsilon * params.epsilon), 1.f);
    else
        clusters.lambda = 1.f;

    // Cluster centers from a host-side RNG (identical in both benchmarks), drawn
    // over the FULL cube on purpose: a center that lands near a wall gets its blob
    // sliced flat by the [-1,1] clamp, which makes the bounding cube visible in the
    // render (flat cut faces on otherwise round blobs).
    std::mt19937 gen(params.seed);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::vector<float3> centers(clusters.k);
    for (auto& c : centers) { c.x = dist(gen); c.y = dist(gen); c.z = dist(gen); }

    float3* d_centers = nullptr;
    cudaMalloc(&d_centers, sizeof(float3) * clusters.k);
    cudaMemcpy(d_centers, centers.data(), sizeof(float3) * clusters.k,
        cudaMemcpyHostToDevice);
    clusters.centers = d_centers;

    unsigned int* d_ids = nullptr;
    cudaMalloc(&d_ids, sizeof(unsigned int) * (size_t)params.count);
    clusters.ids = d_ids;

    return clusters;
}

void destroyClusters(ClusterData& clusters)
{
    cudaFree(clusters.centers);
    cudaFree(clusters.ids);
    clusters.centers = nullptr;
    clusters.ids     = nullptr;
}

size_t clusterBytes(const ClusterData& clusters, size_t point_count)
{
    return sizeof(float3) * clusters.k + sizeof(unsigned int) * point_count;
}

void launchInitPositions(float* pos3, const PointsParams& params,
                         ClusterData& clusters, RngStates& rng)
{
    initPosKernel<<<rng.grid, rng.block>>>(
        pos3, params.count, (const float3*)clusters.centers,
        (unsigned int*)clusters.ids, clusters.k, params.epsilon,
        (curandState*)rng.states);
    cudaDeviceSynchronize();
}

void launchIntegrate3D(float* pos3, size_t point_count, const ClusterData& clusters,
                       RngStates& rng, cudaStream_t s)
{
    integrate3dKernel<<<rng.grid, rng.block, 0, s>>>(
        pos3, point_count, (const float3*)clusters.centers,
        (const unsigned int*)clusters.ids, clusters.lambda, (curandState*)rng.states);
}
