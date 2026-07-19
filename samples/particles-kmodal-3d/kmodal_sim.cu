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

// Spread the low 10 bits of x so each occupies every 3rd bit (Morton/Z-order interleave helper);
// supports grid resolutions up to 1024 per axis (30-bit code, fits uint32).
__device__ __forceinline__ uint32_t sortPart1by2(uint32_t x)
{
    x &= 0x3ffu;
    x = (x | (x << 16)) & 0x030000FFu;
    x = (x | (x <<  8)) & 0x0300F00Fu;
    x = (x | (x <<  4)) & 0x030C30C3u;
    x = (x | (x <<  2)) & 0x09249249u;
    return x;
}
// Morton (Z-order) key of a particle's cell at grid resolution N. Unlike a row-major index it is
// HIERARCHICAL: sorting by it groups particles at every coarser power-of-2 resolution at once, so one
// sort serves any power-of-2 LOD grid <= N without re-coupling the sort to a specific LOD size. The
// key is a pure function of the position -- computed on the fly here, never stored (no O(n) array).
// Codes lie in [0, R^3) where R = next-pow2(N) (the interleave bit width); createSortScratch sizes the
// histogram to R^3 to match.
__device__ __forceinline__ uint32_t sortCellId(float px, float py, float pz, uint32_t N)
{
    float n=(float)N;
    uint32_t cx=(uint32_t)fminf(fmaxf((px+1.f)*0.5f*n,0.f),n-1.f);
    uint32_t cy=(uint32_t)fminf(fmaxf((py+1.f)*0.5f*n,0.f),n-1.f);
    uint32_t cz=(uint32_t)fminf(fmaxf((pz+1.f)*0.5f*n,0.f),n-1.f);
    return sortPart1by2(cx) | (sortPart1by2(cy) << 1) | (sortPart1by2(cz) << 2);
}
// ---- Hand-made onesweep LSB radix sort (uint32 Morton key, uint32 value = particle index) --------
// Reorders particles by sorting (key, index) then gathering pos+ids. No global atomics in the scatter
// (warp __match_any_sync + decoupled look-back, a lock-free chained scan); uint64 internal offsets/
// status so counts are correct past 2^32 (the value index is uint32 -> up to ~4.29B particles, which
// is also where the sort's shadow memory runs out). Resolution-independent: a fixed 256-bucket
// histogram per pass, unlike a counting sort's O(N^3) grid histogram. See scratchpad radix.cu (bench
// vs CUB: ~1.5-1.8x CUB with 64-bit values). Passes = ceil(3*log2(nextpow2(sortN)) / 8).
static constexpr int kRadixBits=8, kRadix=256, kRWarps=kSortThreads/32, kOsItems=8, kOsTile=kSortThreads*kOsItems;
static constexpr unsigned long long kFAgg=1ull<<62, kFPref=2ull<<62, kFMask=3ull<<62;
__device__ __forceinline__ uint32_t rLaneLt(){ uint32_t m; asm("mov.u32 %0, %%lanemask_lt;":"=r"(m)); return m; }

__global__ void keyInitKernel(const float* pos, uint32_t* keys, uint32_t* vals, uint64_t count, uint32_t N){
    for(uint64_t i=(uint64_t)blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=(uint64_t)blockDim.x*gridDim.x){
        keys[i]=sortCellId(pos[i*3],pos[i*3+1],pos[i*3+2],N); vals[i]=(uint32_t)i; } }
// one pass over keys -> per-digit global counts for all 4 digit positions (order-independent)
__global__ void rGlobalHist(const uint32_t* keys, uint64_t n, unsigned long long* gHist){
    __shared__ uint32_t s[4*kRadix];
    for(int r=threadIdx.x;r<4*kRadix;r+=blockDim.x) s[r]=0u; __syncthreads();
    for(uint64_t i=(uint64_t)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(uint64_t)blockDim.x*gridDim.x){
        uint32_t k=keys[i];
        atomicAdd(&s[(k&255)],1u); atomicAdd(&s[kRadix+((k>>8)&255)],1u);
        atomicAdd(&s[2*kRadix+((k>>16)&255)],1u); atomicAdd(&s[3*kRadix+((k>>24)&255)],1u); }
    __syncthreads();
    for(int r=threadIdx.x;r<4*kRadix;r+=blockDim.x) if(s[r]) atomicAdd(&gHist[r],(unsigned long long)s[r]); }
__global__ void rScanGlobal(const unsigned long long* gHist, unsigned long long* gOff){
    int p=blockIdx.x,tid=threadIdx.x; __shared__ unsigned long long s[kRadix];
    s[tid]=gHist[p*kRadix+tid]; __syncthreads();
    for(int o=1;o<kRadix;o<<=1){ unsigned long long t=(tid>=o)?s[tid-o]:0ull; __syncthreads(); s[tid]+=t; __syncthreads(); }
    gOff[p*kRadix+tid]=s[tid]-gHist[p*kRadix+tid]; }
// one kernel per digit: register-held tile + fused local histogram + decoupled look-back, then shared
// local-sort + coalesced write. Grid == numTiles; each block grabs a monotonic tile id (forward progress).
__global__ void rScatter(const uint32_t* keys, const uint32_t* vals, uint64_t n, int sh,
    const unsigned long long* gOff, volatile unsigned long long* status, unsigned int* tileCtr,
    uint64_t numTiles, uint32_t* outKeys, uint32_t* outVals){
    constexpr int TILEN=kOsTile;
    __shared__ uint32_t s_tileId, localHist[kRadix], localBase[kRadix], runOff[kRadix], wd[kRWarps][kRadix], sTot[kRadix];
    __shared__ unsigned long long tileBase[kRadix];
    extern __shared__ char smem[];
    uint32_t* sKey=(uint32_t*)smem; uint32_t* sVal=(uint32_t*)(smem+(size_t)TILEN*4);
    int tid=threadIdx.x,warp=tid>>5,lane=tid&31;
    if(tid==0) s_tileId=atomicAdd(tileCtr,1u); __syncthreads();
    uint32_t tileId=s_tileId; uint64_t base=(uint64_t)tileId*TILEN;
    uint32_t cnt=(base+TILEN<=n)?TILEN:(base<n?(uint32_t)(n-base):0u);
    uint32_t rk[kOsItems],rv[kOsItems],rd[kOsItems];
    #pragma unroll
    for(int s=0;s<kOsItems;s++){ uint64_t i=base+(uint64_t)s*kSortThreads+tid; if(i<n){rk[s]=keys[i];rv[s]=vals[i];rd[s]=(rk[s]>>sh)&255;} else rd[s]=0xFFFFFFFFu; }
    localHist[tid]=0u; __syncthreads();
    #pragma unroll
    for(int s=0;s<kOsItems;s++) if(rd[s]!=0xFFFFFFFFu) atomicAdd(&localHist[rd[s]],1u);
    __syncthreads();
    { uint32_t c=localHist[tid]; localBase[tid]=c; __syncthreads();
      for(int o=1;o<kRadix;o<<=1){ uint32_t t=(tid>=o)?localBase[tid-o]:0u; __syncthreads(); localBase[tid]+=t; __syncthreads(); }
      uint32_t incl=localBase[tid]; __syncthreads(); localBase[tid]=incl-c; runOff[tid]=0u; }
    __syncthreads();
    uint32_t lh=localHist[tid];
    __threadfence(); status[(uint64_t)tileId*kRadix+tid]=kFAgg|(unsigned long long)lh; __threadfence();
    unsigned long long excl=0ull;
    for(int pred=(int)tileId-1;pred>=0;){ unsigned long long v=status[(uint64_t)pred*kRadix+tid]; unsigned long long f=v&kFMask;
        if(f==0ull) continue; excl+=(v&~kFMask); if(f==kFPref) break; --pred; }
    __threadfence(); status[(uint64_t)tileId*kRadix+tid]=kFPref|(excl+(unsigned long long)lh);
    tileBase[tid]=gOff[tid]+excl; __syncthreads();
    for(int s=0;s<kOsItems;s++){
        for(int r=tid;r<kRWarps*kRadix;r+=kSortThreads) ((uint32_t*)wd)[r]=0u; __syncthreads();
        uint32_t d=rd[s]; int rankInWarp=0; unsigned active=__ballot_sync(0xFFFFFFFFu, d!=0xFFFFFFFFu);
        if(d!=0xFFFFFFFFu){ unsigned same=__match_any_sync(active,d)&active; rankInWarp=__popc(same&rLaneLt()); if(lane==(__ffs(same)-1)) wd[warp][d]=__popc(same); }
        __syncthreads();
        for(int dd=tid;dd<kRadix;dd+=kSortThreads){ uint32_t run=0; for(int w=0;w<kRWarps;++w){ uint32_t cc=wd[w][dd]; wd[w][dd]=run; run+=cc; } sTot[dd]=run; }
        __syncthreads();
        if(d!=0xFFFFFFFFu){ uint32_t p=localBase[d]+runOff[d]+wd[warp][d]+rankInWarp; sKey[p]=rk[s]; sVal[p]=rv[s]; }
        __syncthreads();
        for(int dd=tid;dd<kRadix;dd+=kSortThreads) runOff[dd]+=sTot[dd]; __syncthreads();
    }
    for(uint32_t p=tid;p<cnt;p+=kSortThreads){ uint32_t k=sKey[p]; uint32_t d=(k>>sh)&255; uint64_t dest=tileBase[d]+(p-localBase[d]); outKeys[dest]=k; outVals[dest]=sVal[p]; }
}
// gather pos+ids into the sorted order given by the sorted value (index) array
__global__ void gatherKernel(const float* pos, const unsigned int* ids, const uint32_t* order, uint64_t count, float* posOut, unsigned int* idsOut){
    for(uint64_t j=(uint64_t)blockIdx.x*blockDim.x+threadIdx.x;j<count;j+=(uint64_t)blockDim.x*gridDim.x){
        uint64_t src=order[j]; posOut[3*j]=pos[3*src]; posOut[3*j+1]=pos[3*src+1]; posOut[3*j+2]=pos[3*src+2]; idsOut[j]=ids[src]; } }

static int sortKeyBits(uint32_t sortN){ uint32_t R=1; int bits=0; while(R<sortN){ R<<=1; ++bits; } int kb=3*bits; return kb<kRadixBits?kRadixBits:kb; }

SortScratch createSortScratch(size_t count, uint32_t sortN)
{
    SortScratch s; s.sortN=sortN; s.count=count; s.keyBits=sortKeyBits(sortN);
    s.numTiles=(count+kOsTile-1)/kOsTile; s.dynSmem=(size_t)kOsTile*(4+4);   // sKey(u32)+sVal(u32) per elem
    cudaMalloc(&s.keysA,count*4); cudaMalloc(&s.keysB,count*4);
    cudaMalloc(&s.valsA,count*4); cudaMalloc(&s.valsB,count*4);
    cudaMalloc(&s.status,(uint64_t)s.numTiles*kRadix*sizeof(unsigned long long));
    cudaMalloc(&s.gHist,4*kRadix*sizeof(unsigned long long)); cudaMalloc(&s.gOff,4*kRadix*sizeof(unsigned long long));
    cudaMalloc(&s.tileCtr,sizeof(unsigned int));
    cudaMalloc(&s.pos_sorted,count*3*sizeof(float)); cudaMalloc(&s.ids_sorted,count*sizeof(unsigned int));
    cudaFuncSetAttribute(rScatter,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)s.dynSmem);
    return s;
}
void destroySortScratch(SortScratch& s)
{
    cudaFree(s.keysA); cudaFree(s.keysB); cudaFree(s.valsA); cudaFree(s.valsB);
    cudaFree(s.status); cudaFree(s.gHist); cudaFree(s.gOff); cudaFree(s.tileCtr);
    cudaFree(s.pos_sorted); cudaFree(s.ids_sorted);
    s = SortScratch{};
}
size_t sortScratchBytes(const SortScratch& s)
{
    return s.count*(2*4 + 2*4 + 3*sizeof(float) + sizeof(unsigned int))      // keys+vals ping-pong + pos/ids shadow
         + (uint64_t)s.numTiles*kRadix*sizeof(unsigned long long)             // look-back status
         + (4*kRadix*2)*sizeof(unsigned long long) + sizeof(unsigned int);    // gHist/gOff/tileCtr
}
void launchSpatialSort(float* pos3, unsigned int* ids, size_t count, SortScratch& s, cudaStream_t stream)
{
    auto* kA=(uint32_t*)s.keysA; auto* kB=(uint32_t*)s.keysB; auto* vA=(uint32_t*)s.valsA; auto* vB=(uint32_t*)s.valsB;
    auto* status=(unsigned long long*)s.status; auto* gHist=(unsigned long long*)s.gHist; auto* gOff=(unsigned long long*)s.gOff;
    auto* tileCtr=(unsigned int*)s.tileCtr; auto* posS=(float*)s.pos_sorted; auto* idsS=(unsigned int*)s.ids_sorted;
    static int sm=0; if(!sm){ int d; cudaGetDevice(&d); cudaDeviceGetAttribute(&sm,cudaDevAttrMultiProcessorCount,d); }
    uint32_t partBlocks=sortGridFor(count); if(partBlocks>(uint32_t)sm*32) partBlocks=(uint32_t)sm*32;
    const int passes=(s.keyBits+kRadixBits-1)/kRadixBits;
    static const bool ktime=(std::getenv("MIMIR_SIM_SORT_KTIME")!=nullptr);
    cudaEvent_t e[4]{}; if(ktime){ for(auto&ev:e) cudaEventCreate(&ev); cudaEventRecord(e[0],stream); }
    keyInitKernel<<<partBlocks,kSortThreads,0,stream>>>(pos3,kA,vA,count,s.sortN);
    cudaMemsetAsync(gHist,0,4*kRadix*sizeof(unsigned long long),stream);
    rGlobalHist<<<(uint32_t)sm*16,kSortThreads,0,stream>>>(kA,count,gHist);
    rScanGlobal<<<4,kRadix,0,stream>>>(gHist,gOff);
    uint32_t* ksrc=kA; uint32_t* vsrc=vA; uint32_t* kdst=kB; uint32_t* vdst=vB;
    for(int p=0;p<passes;p++){
        cudaMemsetAsync(status,0,(uint64_t)s.numTiles*kRadix*sizeof(unsigned long long),stream);
        cudaMemsetAsync(tileCtr,0,sizeof(unsigned int),stream);
        rScatter<<<(uint32_t)s.numTiles,kSortThreads,s.dynSmem,stream>>>(
            ksrc,vsrc,count,p*kRadixBits,gOff+(uint64_t)p*kRadix,status,tileCtr,s.numTiles,kdst,vdst);
        std::swap(ksrc,kdst); std::swap(vsrc,vdst);
    }
    if(ktime) cudaEventRecord(e[1],stream);
    gatherKernel<<<partBlocks,kSortThreads,0,stream>>>(pos3,ids,vsrc,count,posS,idsS); // vsrc = sorted indices
    if(ktime) cudaEventRecord(e[2],stream);
    cudaMemcpyAsync(pos3,posS,count*3*sizeof(float),cudaMemcpyDeviceToDevice,stream);
    cudaMemcpyAsync(ids,idsS,count*sizeof(unsigned int),cudaMemcpyDeviceToDevice,stream);
    if(ktime){ cudaEventRecord(e[3],stream); cudaEventSynchronize(e[3]);
        float t1,t2,t3; cudaEventElapsedTime(&t1,e[0],e[1]); cudaEventElapsedTime(&t2,e[1],e[2]); cudaEventElapsedTime(&t3,e[2],e[3]);
        std::fprintf(stderr,"[sim-sort-ktime] keyinit+radix(%d passes) %.2f | gather %.2f | copyback %.2f ms\n",passes,t1,t2,t3);
        for(auto&ev:e) cudaEventDestroy(ev); }
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
