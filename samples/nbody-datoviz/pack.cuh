#pragma once

#include <vector_types.h> // float4

// Compacts the simulation's float4 positions (xyz + mass) into a tightly packed
// vec3 (xyz) device buffer that matches datoviz's `dvz_marker_position` layout.
//
// Doing this on the GPU before the device->host copy means we transfer only 3 floats
// per body across PCIe (instead of 4) and avoid an O(n) host-side repack every frame,
// which is the smallest data movement possible given datoviz cannot read the CUDA
// buffer directly.
void packPositionsVec3(const float4 *d_pos4, float *d_pos3, unsigned int body_count,
    int block_size
);
