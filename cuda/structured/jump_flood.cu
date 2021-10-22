#include "jump_flood.hpp"
#include "cuda_utils.hpp"

#include <chrono>
#include <random>


__device__ float distance2D(float2 a, float2 b, int width, int height)
{
  return hypotf(b.x - a.x, b.y - a.y) / hypotf(width, height);
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

	/**
	 * 1. Generate some seed pixel locations.
	 */
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

	/**
	 * 2. Set the R,G values of the seed pixels to their own tex coordinates.
	 */
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
					printf("New seed at %f %f\n", voronoiSeeds[iSeed].x, voronoiSeeds[iSeed].y);
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
  checkCuda(cudaMalloc(&_d_distances, dist_size));
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
  checkCuda(cudaFree(_d_distances));
  checkCuda(cudaDeviceReset());
}

void JumpFloodProgram::runTimestep()
{
  // Define dimensions & launch kernel
	dim3 block_dims(_block_size, _block_size, 1);
	dim3 grid_dims(_extent.x / block_dims.x, _extent.y / block_dims.y, 1);
	jumpFloodKernel<<<grid_dims, block_dims>>>(
    _d_distances, _d_grid, _extent.x, _extent.y
  );
	checkCuda(cudaDeviceSynchronize());
}
