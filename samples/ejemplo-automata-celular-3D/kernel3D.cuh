#pragma once

__device__ int arrayIndex(int x, int y, int z, int n)
{
    return x + (y * n) + (z * n * n);
}

__global__ void kernel_CA3D(int n, int *in, int *out){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= n || y >= n || z >= n)
        return;

    int right = x + 1 == n ? 0 : x + 1;
    int left = x - 1 < 0 ? n - 1 : x - 1;

    int forward = y + 1 == n ? 0 : y + 1;
    int backward = y - 1 < 0 ? n - 1 : y - 1;

    int up = z + 1 == n ? 0 : z + 1;
    int down = z - 1 < 0 ? n - 1 : z - 1;

    int self = in[arrayIndex(x, y, z, n)];

    /**
     * 0 1 2 | 8  9  10 | 17 18 19
     * 3 X 4 | 11 12 13 | 20 21 22
     * 5 6 7 | 14 15 16 | 23 24 25
     */

    // SAME LEVEL

    int neighbor0 = in[arrayIndex(left, forward, z, n)];
    int neighbor1 = in[arrayIndex(x, forward, z, n)];
    int neighbor2 = in[arrayIndex(right, forward, z, n)];

    int neighbor3 = in[arrayIndex(left, y, z, n)];
    int neighbor4 = in[arrayIndex(right, y, z, n)];

    int neighbor5 = in[arrayIndex(left, backward, z, n)];
    int neighbor6 = in[arrayIndex(x, backward, z, n)];
    int neighbor7 = in[arrayIndex(right, backward, z, n)];

    int neighbors_0 = neighbor0 + neighbor1 + neighbor2 + neighbor3 + neighbor4 + neighbor5 + neighbor6 + neighbor7;

    // UP

    int neighbor8 = in[arrayIndex(left, forward, up, n)];
    int neighbor9 = in[arrayIndex(x, forward, up, n)];
    int neighbor10 = in[arrayIndex(right, forward, up, n)];

    int neighbor11 = in[arrayIndex(left, y, up, n)];
    int neighbor12 = in[arrayIndex(x, y, up, n)];
    int neighbor13 = in[arrayIndex(right, y, up, n)];

    int neighbor14 = in[arrayIndex(left, backward, up, n)];
    int neighbor15 = in[arrayIndex(x, backward, z, up)];
    int neighbor16 = in[arrayIndex(right, backward, up, n)];

    int neighbors_1 = neighbor8 + neighbor9 + neighbor10 + neighbor11 + neighbor12 + neighbor13 + neighbor14 + neighbor15 + neighbor16;

    // DOWN

    int neighbor17 = in[arrayIndex(left, forward, down, n)];
    int neighbor18 = in[arrayIndex(x, forward, down, n)];
    int neighbor19 = in[arrayIndex(right, forward, down, n)];

    int neighbor20 = in[arrayIndex(left, y, down, n)];
    int neighbor21 = in[arrayIndex(x, y, down, n)];
    int neighbor22 = in[arrayIndex(right, y, down, n)];

    int neighbor23 = in[arrayIndex(left, backward, down, n)];
    int neighbor24 = in[arrayIndex(x, backward, down, n)];
    int neighbor25 = in[arrayIndex(right, backward, down, n)];

    int neighbors_2 = neighbor17 + neighbor18 + neighbor19 + neighbor20 + neighbor21 + neighbor22 + neighbor23 + neighbor24 + neighbor25;

    int neighbors = neighbors_0 + neighbors_1 + neighbors_2;
    out[arrayIndex(x, y, z, n)] = self ? (neighbors == CA_LOW || neighbors == CA_HIGH ? 1 : 0) : (neighbors == CA_NACER ? 1 : 0);
}

