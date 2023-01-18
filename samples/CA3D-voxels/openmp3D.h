#pragma once
void openmp_CA3D(int n, int *in, int *out){
    // loops para explorar el cubo de (n x n x n)
    long up, down, left, right, back, front;
    #pragma omp parallel for
    for(int z=0; z<n; ++z){ 
        for(int y=0; y<n; ++y){
            for(int x=0; x<n; ++x){
               up    = y-1 < 0 ? n-1 : y-1;
               down  = (y+1) % n;

               left  = x-1 < 0 ? n-1 : x-1;
               right = (x+1) % n;

               back  = z-1 < 0 ? n-1 : z-1;
               front = (z+1) % n;

               // recorrer el mini-cubo de vecindad por capas de Z
               // capa trasera --> 9 vecinos
               int vBack  = in[back*n*n + up*n + left]   + in[back*n*n + up*n + x]    + in[back*n*n + up*n   + right]   +
                                  in[back*n*n + y*n  + left]   + in[back*n*n + y*n  + x]    + in[back*n*n + y*n    + right]    + 
                                  in[back*n*n + down*n + left] + in[back*n*n + down*n  + x] + in[back*n*n + down*n + right];

               // capa intermedia --> 8 vecinos
               int vMid   = in[z*n*n + up*n + left]   + in[z*n*n + up*n + x]    + in[z*n*n + up*n   + right] +
                                  in[z*n*n + y*n  + left]   + 0                       + in[z*n*n + y*n    + right] + 
                                  in[z*n*n + down*n + left] + in[z*n*n + down*n  + x] + in[z*n*n + down*n + right];

               // capa frontal --> 9 vecinos
               int vFront = in[front*n*n + up*n + left]   + in[front*n*n + up*n + x]     + in[front*n*n + up*n   + right] +
                                  in[front*n*n + y*n  + left]   + in[front*n*n + y*n  + x]     + in[front*n*n + y*n    + right] + 
                                  in[front*n*n + down*n + left] + in[front*n*n + down*n + x]   + in[front*n*n + down*n + right];

               int vecinos = vBack + vMid + vFront;
               int celda = in[z*n*n + y*n + x];
               if(celda == 1 && (vecinos == CA_LOW || vecinos == CA_HIGH)){
                    out[z*n*n + y*n + x] = 1;
               }
               else if(celda == 0 && vecinos == CA_NACER){
                    out[z*n*n + y*n + x] = 1;
               }
               else{
                    out[z*n*n + y*n + x] = 0;
               }
            }
        }
    }
}
