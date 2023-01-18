#pragma once
#include <random>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}



void print_array(int n, long *array, const char *msg){
    if(n > 32){
       return;
    }
    printf("%s[", msg);
    for(int i=0; i<n-1; ++i){
        printf("%ld, ", array[i]);
    }
    if(n>0){
        printf("%ld", array[n-1]);
    }
    printf("]\n");
}


void print_mat(int n, int *m, const char* msg){
    if(n > 128){
        return;
    }
    const char *map[2] = {" ", "*"};
    printf("%s\n", msg); fflush(stdout);
    for(int i=0; i<n; ++i){
        for(int j=0; j<n; ++j){
            printf("%s ", map[m[i*n + j]]);
            //printf("%i ", m[i*n + j]);
        }
        printf("\n");
    }
}


void print_cube(int n, int *m, const char* msg){
    if(n > 16){
        return;
    }
    const char *map[2] = {" ", "*"};
    printf("%s\n", msg); fflush(stdout);
    for(int z=0; z<n; ++z){
        printf("\n Layer Z=%i\n", z);
        for(int y=0; y<n; ++y){
            for(int x=0; x<n; ++x){
               printf("%s ", map[m[z*n*n + y*n + x]]);
            }
            printf("\n");
        }
    }
}

void init_prob(int n, int *m, int seed, float prob){
    #pragma omp parallel shared(m)
    {
        int tid = omp_get_thread_num();
        int nt = omp_get_num_threads();
        long elems = (long)n * (long)n * (long)n;
        long chunk = elems/nt;
        long start = tid*chunk;
        long end = start + chunk;
        std::mt19937 mt(seed+tid);
        std::uniform_real_distribution<float>  dist(0, 1);
        for(int k=start; k<elems && k<end; ++k){
            float val = dist(mt);
            m[k] = val <= prob? 1 : 0;
        }
    }
}
