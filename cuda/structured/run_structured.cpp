#include "jump_flood.hpp"
#include "cudaview/vk_cuda_engine.hpp"

#include <iostream>
#include <cstdio>
#include <chrono>
#include <random>
#include <cmath>

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

int main(int argc, char *argv[])
{
  size_t point_count = 10;
  if (argc >= 2)
  {
    point_count = std::stoul(argv[1]);
  }

  const int diagramXDim = 256;
	const int diagramYDim = 256;
	const int channels = 3;

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
	float2 voronoiCanvas[diagramXDim * diagramYDim];
	float2* voronoiCanvasFill = voronoiCanvas;
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

	/**
	 * Allocate GPU resources and launch the main computation kernel.
	 */
	unsigned char* voronoiOutputImage = (unsigned char*)malloc(
    diagramXDim * diagramYDim * channels * sizeof(unsigned char)
  );
	jumpFloodWithCuda(voronoiOutputImage, voronoiCanvas, diagramXDim, diagramYDim);

	/**
	 * Cleanup.
	 */
	free(voronoiOutputImage);
	free(voronoiSeedsUV);

  try
  {
    // Initialize engine
    //VulkanCudaEngine engine(program._particle_count, program._stream);
    //engine.init(800, 600);
    //engine.registerDeviceMemory(program._d_coords);

    // Cannot make CUDA calls that use the target device memory before
    // registering it on the engine
    //program.setInitialState();

    // Set up the function that we want to display
    //auto timestep_function = std::bind(&CudaProgram::runTimestep, program);
    //engine.registerFunction(timestep_function, iter_count);

    // Start rendering loop
    //engine.mainLoop();
  }
  catch (const std::exception& e)
  {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

//VulkanEngine engine(vertices.size());
