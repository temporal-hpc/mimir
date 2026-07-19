#include "mimir/voxel_lod.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <random>
#include <algorithm>

static int refCell(const std::vector<int>& f, uint32_t N, uint32_t M, uint32_t cx, uint32_t cy, uint32_t cz){
    auto lo=[&](uint32_t c){return (uint32_t)((uint64_t)c*N/M);}; auto hi=[&](uint32_t c){return (uint32_t)(((uint64_t)c+1)*N/M);};
    int m=0; for(uint32_t z=lo(cz);z<hi(cz);++z)for(uint32_t y=lo(cy);y<hi(cy);++y)for(uint32_t x=lo(cx);x<hi(cx);++x)
        m=std::max(m,f[(size_t)x+N*((size_t)y+(size_t)N*z)]); return m;
}
int main(){
    struct C{uint32_t N,M;} cs[]={{8,4},{9,4},{16,4},{100,25},{128,32},{130,31}};
    std::mt19937 rng(7);
    for(auto c:cs){
        uint64_t nc=(uint64_t)c.N*c.N*c.N, mc=(uint64_t)c.M*c.M*c.M;
        std::vector<int> f(nc); std::uniform_int_distribution<int> d(0,3); for(auto&v:f)v=d(rng);
        int *df,*dc; cudaMalloc(&df,nc*4); cudaMalloc(&dc,mc*4); cudaMemcpy(df,f.data(),nc*4,cudaMemcpyHostToDevice);
        mimir::voxelPoolMax(df,c.N,dc,c.M,0); cudaDeviceSynchronize();
        std::vector<int> g(mc); cudaMemcpy(g.data(),dc,mc*4,cudaMemcpyDeviceToHost);
        uint64_t bad=0; for(uint32_t z=0;z<c.M;++z)for(uint32_t y=0;y<c.M;++y)for(uint32_t x=0;x<c.M;++x)
            if(g[(size_t)x+c.M*((size_t)y+(size_t)c.M*z)]!=refCell(f,c.N,c.M,x,y,z)) bad++;
        printf("[pool N=%u M=%u] %s\n",c.N,c.M,bad?"FAIL":"OK"); if(bad) return 1;
        cudaFree(df); cudaFree(dc);
    }
    printf("voxel_pool_ref_test: ALL PASS\n"); return 0;
}
