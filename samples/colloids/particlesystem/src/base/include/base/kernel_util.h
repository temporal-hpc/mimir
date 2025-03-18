/*
 * kernel_util.cuh
 *
 *  Created on: 22-07-2016
 *      Author: francisco
 */

#ifndef KERNEL_UTIL_CUH_
#define KERNEL_UTIL_CUH_

namespace particlesystem {

template<typename T>
void memsetArray(T *d_arr, const T value, const size_t array_size,
                 uint2 launchData);

template<typename T>
void memsetDualArrays(T *d_arr1, T *d_arr2, const T value, const size_t n,
                      uint2 launchData);


void copyPositions(double2 *d_dst, double2 *d_src, const size_t array_size,
                   uint2 launchData);

} // namespace particlesystem

#endif /* KERNEL_UTIL_CUH_ */
