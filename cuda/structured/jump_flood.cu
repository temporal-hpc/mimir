#include "jump_flood.hpp"
#include "cuda_utils.hpp"

#include <chrono>
#include <limits> // std::numeric_limits
#include <random>

constexpr float max_distance = std::numeric_limits<float>::max();

__device__ int clamp(int x, int a, int b)
{
    return max(a, min(b, x));
}

__device__ float distance2D(float2 a, float2 b, int width, int height)
{
  return hypotf((b.x - a.x) / width, (b.y - a.y) / height);
}

__device__ float3 jumpFloodStep(const float2 coord, const float3 *seeds,
  const int step_length, const int2 extent)
{
  float best_dist = max_distance;
  float2 best_coord{0.f, 0.f};


  return make_float3(best_coord.x, best_coord.y, best_dist);
}

__global__
void kernelJumpFlood(float *distances, float2 *seeds, const int2 extent)
{
  const int tx = threadIdx.x + blockIdx.x * blockDim.x;
  const int ty = threadIdx.y + blockIdx.y * blockDim.y;
  // TODO: Handle boundaries
  if (tx < extent.x && ty < extent.y)
  {
    auto self_idx = ty * extent.x + tx;

    auto max_extent = max(extent.x, extent.y);
    for (int step = max_extent / 2; step > 0; step = step >> 1)
    {
      auto own_seed = seeds[self_idx];
      float2 best_coord{own_seed.x, own_seed.y};
      auto best_dist = max_distance;
      for (int y = -1; y <= 1; ++y)
      {
        for (int x = -1; x <= 1; ++x)
        {
          auto lookup_x = tx + x * step;
          auto lookup_y = ty + y * step;
          if (lookup_x < 0 || lookup_x > extent.x || lookup_y < 0 || lookup_y > extent.y)
          {
            continue;
          }
          auto seed = seeds[extent.y * lookup_y + lookup_x];
          auto dist = hypotf(seed.x - tx, seed.y - ty);

          if ((seed.x != 0.f || seed.y != 0.f) && dist < best_dist)
          {
            best_dist = dist;
            best_coord = make_float2(seed.x, seed.y);
          }
        }
      }
      __syncthreads();
      distances[self_idx] = best_dist;
      seeds[self_idx] = best_coord;
    }
    auto extent_distance = hypotf(extent.x, extent.y);
    distances[self_idx] /= extent_distance;
  }
}

void jumpFlood(float* distances, float2 *seeds, int2 extent)
{
  dim3 block(32, 32);
  dim3 grid( (extent.x + block.x - 1) / block.x, (extent.y + block.y - 1) / block.y );
  kernelJumpFlood<<< grid, block >>>(distances, seeds, extent);
  checkCuda(cudaDeviceSynchronize());
}

__global__ void jumpFloodKernel(float *distances, float2 *seeds, int diagramXDim, int diagramYDim)
{
	// calculate non-normalized texture coordinates
	unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned y = blockIdx.y * blockDim.y + threadIdx.y;

	// Ignore out-of-bounds index
	if (x > diagramXDim || y > diagramYDim) return;

	float maximalDim = fmaxf(diagramXDim, diagramYDim);

	// JFA pass(es) loop
	for (int passIndex = 0; passIndex < log2f(maximalDim); ++passIndex)
	{
		float step = powf(2.f, (log2f(maximalDim) - passIndex - 1.f));

		// At first, the best candidate is ourselves
		unsigned selfIdx = y * diagramXDim + x;
		float2 closestCandidate = seeds[selfIdx];
		float closestDistance = float(INT_MAX);

		// JFA pass computations
		for (int gridY = 0; gridY < 3; ++gridY)
		{
			for (int gridX = 0; gridX < 3; ++gridX)
			{
				float xLookup = x - step + gridX * step;
				float yLookup = y - step + gridY * step;

				// Ignore out-of-bounds
				if (xLookup < 1e-6f || xLookup > diagramXDim || yLookup < 1e-6f || yLookup > diagramYDim) continue;

				int lookupIdx = yLookup * diagramXDim + xLookup;
				float2 otherCandidate = seeds[lookupIdx];

				if (otherCandidate.x + otherCandidate.y > 1e-6f)
				{
					float otherDistance = sqrtf(
						(otherCandidate.x - x) * (otherCandidate.x - x)
						 + (otherCandidate.y - y) * (otherCandidate.y - y)
					);
					if (otherDistance < closestDistance)
					{
						closestCandidate = otherCandidate;
						closestDistance = otherDistance;
					}
				}
			}
		}

		distances[selfIdx] = closestDistance;
		seeds[selfIdx] = closestCandidate;

		__syncthreads();
	}
  //if (x < 5 && y < 5) printf("%d %d %f\n", x, y, distances[y * diagramXDim + x]);
}

float2* randomUniformClamped2D(size_t nPoints)
{
	std::mt19937_64 goodRng;

	// Time dependent seed, the "modern" way
	uint64_t genSeed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
	std::seed_seq sequence{ uint32_t(genSeed & 0xFFFFFFFF), uint32_t(genSeed >> 32) };
	goodRng.seed(sequence);

	// Initialize uniform distribution, clamped to [0,1)
	std::uniform_real_distribution<float> unif(0, 1);

	// Generate point "cloud"
	float2* points = (float2*)malloc(nPoints * sizeof(float2));
	for (int iSim = 0; iSim < nPoints; ++iSim)
	{
		points[iSim].x = unif(goodRng);
		points[iSim].y = unif(goodRng);
	}

	return points;
}

JumpFloodProgram::JumpFloodProgram(size_t point_count, int width, int height):
  _element_count{point_count}, _extent{width, height},
  _grid_size((_element_count + _block_size - 1) / _block_size)
{
  checkCuda(cudaStreamCreateWithFlags(&_stream, cudaStreamNonBlocking));
}

void JumpFloodProgram::setInitialState()
{
  auto point_count = _element_count;
  const int diagramXDim = _extent.x;
	const int diagramYDim = _extent.y;

	float2* voronoiSeedsUV = randomUniformClamped2D(point_count);
	// Texture coordinates (sub-pixel measurement, similar to gl_TexCoord)
	float2 voronoiSeeds[point_count];
	for (int iSeed = 0; iSeed < point_count; ++iSeed)
	{
		voronoiSeeds[iSeed] = float2{
			.49f + floor(voronoiSeedsUV[iSeed].x * float(diagramXDim)),
			.49f + floor(voronoiSeedsUV[iSeed].y * float(diagramYDim))
		};
	}

	float2 hostNumericCanvas[diagramXDim * diagramYDim];
	float2* voronoiCanvasFill = hostNumericCanvas;
	for (int iRow = 0; iRow < diagramYDim; iRow++)
	{
		for (int iCol = 0; iCol < diagramXDim; iCol++)
		{
			for (int iSeed = 0; iSeed < point_count; ++iSeed)
			{
				if (iCol == int(voronoiSeeds[iSeed].x) && iRow == int(voronoiSeeds[iSeed].y))
				{
					//printf("New seed at %f %f\n", voronoiSeeds[iSeed].x, voronoiSeeds[iSeed].y);
					voronoiCanvasFill->x = voronoiSeeds[iSeed].x;
					voronoiCanvasFill->y = voronoiSeeds[iSeed].y;
					goto canvasIterationDone;
				}
			}

			voronoiCanvasFill->x = 0.f;
			voronoiCanvasFill->y = 0.f;

		canvasIterationDone:
			voronoiCanvasFill++;
		}
	}

	checkCuda(cudaSetDevice(0));

	// Allocate device numeric canvas
	size_t numericCanvasSize = sizeof(float2) * diagramXDim * diagramYDim;
	checkCuda(cudaMalloc((void**)&_d_grid, numericCanvasSize));
	checkCuda(cudaMemset(_d_grid, 0, numericCanvasSize));

  auto dist_size = sizeof(float) * diagramXDim * diagramYDim;
  //checkCuda(cudaMalloc(&_d_distances, dist_size));
  checkCuda(cudaMemset(_d_distances, 0, dist_size));

	// Copy into
	checkCuda(cudaMemcpy2D(
		_d_grid,
		_extent.x * sizeof(float2),
		hostNumericCanvas,
		_extent.x * sizeof(float2),
		_extent.x * sizeof(float2),
		_extent.y,
		cudaMemcpyHostToDevice
	));

  free(voronoiSeedsUV);
}

void JumpFloodProgram::cleanup()
{
  checkCuda(cudaStreamSynchronize(_stream));
  checkCuda(cudaStreamDestroy(_stream));
  checkCuda(cudaFree(_d_grid));
  //checkCuda(cudaFree(_d_distances));
  checkCuda(cudaDeviceReset());
}

void JumpFloodProgram::runTimestep()
{
  jumpFlood(_d_distances, _d_grid, _extent);
  checkCuda(cudaDeviceSynchronize());
  /*// Define dimensions & launch kernel
	dim3 block_dims(_block_size, _block_size, 1);
	dim3 grid_dims(_extent.x / block_dims.x, _extent.y / block_dims.y, 1);
	jumpFloodKernel<<<grid_dims, block_dims>>>(
    _d_distances, _d_grid, _extent.x, _extent.y
  );
	checkCuda(cudaDeviceSynchronize());*/
}
