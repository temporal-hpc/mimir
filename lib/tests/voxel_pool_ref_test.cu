#include "mimir/voxel_lod.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <set>
#include <tuple>
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
    // --- voxelCompactLiving: the compacted set of living-cell centers must equal the alive cells. ---
    {
        struct C{uint32_t N; float p;} cs2[]={{8,0.0f},{8,0.5f},{16,0.3f},{32,0.1f},{31,0.7f},{9,1.0f}};
        float3 origin{-1.f, 2.f, 0.5f}, spacing{2.f, 1.f, 0.5f}; // arbitrary mapping
        std::uniform_real_distribution<float> u(0.f,1.f);
        for(auto c:cs2){
            uint64_t nc=(uint64_t)c.N*c.N*c.N;
            std::vector<int> f(nc); uint32_t alive=0;
            for(auto&v:f){ v = (u(rng) < c.p) ? 1 : 0; alive += (uint32_t)v; }
            // expected set of living-cell world centers (rounded to integer grid keys to compare exactly)
            std::set<std::tuple<uint32_t,uint32_t,uint32_t>> expect;
            for(uint64_t i=0;i<nc;++i) if(f[i]){ uint32_t x=(uint32_t)(i%c.N),y=(uint32_t)((i/c.N)%c.N),z=(uint32_t)(i/((uint64_t)c.N*c.N)); expect.insert({x,y,z}); }

            int *df; cudaMalloc(&df,nc*4); cudaMemcpy(df,f.data(),nc*4,cudaMemcpyHostToDevice);
            float *dpos; cudaMalloc(&dpos, nc*3*sizeof(float));
            uint32_t *dcnt; cudaMalloc(&dcnt,sizeof(uint32_t));
            mimir::voxelCompactLiving(df,c.N,origin,spacing,dpos,(uint32_t)nc,dcnt,0); cudaDeviceSynchronize();
            uint32_t cnt=0; cudaMemcpy(&cnt,dcnt,sizeof(uint32_t),cudaMemcpyDeviceToHost);
            std::vector<float> pos((size_t)cnt*3); cudaMemcpy(pos.data(),dpos,(size_t)cnt*3*sizeof(float),cudaMemcpyDeviceToHost);

            bool ok = (cnt==alive);
            std::set<std::tuple<uint32_t,uint32_t,uint32_t>> got;
            for(uint32_t k=0;k<cnt && ok;++k){
                // invert world = origin + (x,y,z)*spacing
                float fx=(pos[3*k+0]-origin.x)/spacing.x, fy=(pos[3*k+1]-origin.y)/spacing.y, fz=(pos[3*k+2]-origin.z)/spacing.z;
                got.insert({(uint32_t)(fx+0.5f),(uint32_t)(fy+0.5f),(uint32_t)(fz+0.5f)});
            }
            if(ok) ok = (got==expect);
            printf("[compact N=%u p=%.2f alive=%u got=%u] %s\n", c.N, c.p, alive, cnt, ok?"OK":"FAIL");
            cudaFree(df); cudaFree(dpos); cudaFree(dcnt);
            if(!ok) return 1;
        }
    }

    printf("voxel_pool_ref_test: ALL PASS\n"); return 0;
}
