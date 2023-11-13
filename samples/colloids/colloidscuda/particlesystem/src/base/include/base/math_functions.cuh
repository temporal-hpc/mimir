/*
 * math_functions.cuh
 *
 *  Created on: Jan 22, 2018
 *      Author: francisco
 */

#ifndef MATH_FUNCTIONS_CUH_
#define MATH_FUNCTIONS_CUH_

#include <math_constants.h> // CUDART:PI, CUDART_PI_F

namespace particlesystem {

// Computes the distance vector between two 2D vectors, using the minimum image
// convention for a square box of length l.
inline __device__ __host__
double2 distVec(double2 a, double2 b, double boxlength)
{
	double2 r = make_double2(b.x - a.x, b.y - a.y);
	double Lhalf = boxlength / 2.0;

	if (r.x > Lhalf) r.x -= boxlength;
	else if (r.x < -Lhalf) r.x += boxlength;
	if (r.y > Lhalf) r.y -= boxlength;
	else if (r.y < -Lhalf) r.y += boxlength;

	return r;
}

// Wraps a position around the 2D box of length l. Does nothing if the position
// is already the box.
inline __device__ __host__
double2 wrapBoundary(double2 pos, double boxlength)
{
	pos.x -= floor(pos.x / boxlength) * boxlength;
	pos.y -= floor(pos.y / boxlength) * boxlength;

	return pos;
}

// Returns the dot product between two 2D vectors.
inline __device__ __host__ double dot(double2 u, double2 v)
{
	return u.x * v.x + u.y * v.y;
}

// Returns the scalar cross product between two 2D vectors, which corresponds to
// the magnitude of the Z-coordinate of the resulting vector.
inline __device__ __host__ double cross(double2 u, double2 v)
{
	return u.x * v.y - v.x * u.y;
}

// Returns the smallest angle between two 2D vectors.
inline __device__ __host__ double angle(double2 u, double2 v)
{
	return atan2(abs(cross(u, v)), dot(u, v)) * 180.0 / CUDART_PI;
}

// Test for intersection between two 2D line segments represented by their
// respective extremes. Returns true if the lines intersect, false otherwise.
inline __device__ __host__
bool lineIntersection(double2 p0, double2 p1, double2 p2, double2 p3, double l)
{
	double2 s1 = distVec(p0, p1, l);
	double2 s2 = distVec(p2, p3, l);
	double2 s3 = distVec(p2, p0, l);
	double det = cross(s1, s2);
	double s = cross(s1, s3) / det;
	double t = cross(s2, s3) / det;

	return (s > 0 && s < 1 && t > 0 && t < 1);
}

} // namespace particlesystem


#endif /* MATH_FUNCTIONS_CUH_ */
