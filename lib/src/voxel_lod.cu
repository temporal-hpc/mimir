#include "mimir/voxel_lod.hpp"
namespace mimir {
namespace { constexpr int kThreads = 256;
__global__ void poolMaxKernel(const int* fine, uint32_t N, int* coarse, uint32_t M){
    uint64_t mCells=(uint64_t)M*M*M;
    for(uint64_t c=(uint64_t)blockIdx.x*blockDim.x+threadIdx.x;c<mCells;c+=(uint64_t)blockDim.x*gridDim.x){
        uint32_t cx=(uint32_t)(c%M), cy=(uint32_t)((c/M)%M), cz=(uint32_t)(c/((uint64_t)M*M));
        uint32_t x0=(uint32_t)((uint64_t)cx*N/M),x1=(uint32_t)(((uint64_t)cx+1)*N/M);
        uint32_t y0=(uint32_t)((uint64_t)cy*N/M),y1=(uint32_t)(((uint64_t)cy+1)*N/M);
        uint32_t z0=(uint32_t)((uint64_t)cz*N/M),z1=(uint32_t)(((uint64_t)cz+1)*N/M);
        int m=0; for(uint32_t z=z0;z<z1;++z)for(uint32_t y=y0;y<y1;++y)for(uint32_t x=x0;x<x1;++x)
            m=max(m,fine[(uint64_t)x+N*((uint64_t)y+(uint64_t)N*z)]);
        coarse[c]=m;
    }
}}
void voxelPoolMax(const int* fine,uint32_t N,int* coarse,uint32_t M,cudaStream_t s){
    uint64_t b=((uint64_t)M*M*M+kThreads-1)/kThreads; if(b>2147483647ull)b=2147483647ull; if(b<1)b=1;
    poolMaxKernel<<<(uint32_t)b,kThreads,0,s>>>(fine,N,coarse,M);
}}
