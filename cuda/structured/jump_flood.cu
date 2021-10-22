#include "cuda_utils.hpp"

__device__ uchar3 toColor(float2 texCoords, int imageW, int imageH)
{
	return uchar3{
    static_cast<unsigned char>(int(255.99f * texCoords.x / float(imageW))),
		static_cast<unsigned char>(int(255.99f * texCoords.y / float(imageH))),
		static_cast<unsigned char>(int(255.99f * .3f))
	};
}


__global__ void jumpFloodKernel(uchar3* pixelCanvas, float2* numericCanvas, int diagramXDim, int diagramYDim)
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
		float2 closestCandidate = numericCanvas[selfIdx];
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
				float2 otherCandidate = numericCanvas[lookupIdx];

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

		pixelCanvas[selfIdx] = toColor(closestCandidate, diagramXDim, diagramYDim);
		numericCanvas[selfIdx] = closestCandidate;

		__syncthreads();
	}
}


void jumpFloodWithCuda(unsigned char* hostPixelCanvas, float2* hostNumericCanvas, int diagramXDim, int diagramYDim)
{
	// For sanity
	checkCuda(cudaSetDevice(0));

	int pixelChannels = 3;
	int numericChannels = 2;

	// Allocate device numeric canvas
	float2* deviceNumericCanvas;
	size_t numericCanvasSize = diagramXDim * diagramYDim * numericChannels * sizeof(float);
	checkCuda(cudaMalloc((void**)&deviceNumericCanvas, numericCanvasSize));
	checkCuda(cudaMemset(deviceNumericCanvas, 0, numericCanvasSize));


	// Allocate device pixel canvas
	uchar3* devicePixelCanvas;
	size_t pixelCanvasSize = diagramXDim * diagramYDim * pixelChannels * sizeof(unsigned char);
	checkCuda(cudaMalloc((void**)&devicePixelCanvas, pixelCanvasSize));
	checkCuda(cudaMemset(devicePixelCanvas, 0, pixelCanvasSize));

	// Copy into
	checkCuda(cudaMemcpy2D(
		deviceNumericCanvas,
		diagramXDim * numericChannels * sizeof(float),
		hostNumericCanvas,
		diagramXDim * numericChannels * sizeof(float),
		diagramXDim * numericChannels * sizeof(float),
		diagramYDim,
		cudaMemcpyHostToDevice
	));


	// Define dimensions & launch kernel
	dim3 bDim(32, 32, 1);
	dim3 gDim(diagramXDim / bDim.x, diagramYDim / bDim.y, 1);
	jumpFloodKernel<<<gDim, bDim>>>(
    devicePixelCanvas, deviceNumericCanvas, diagramXDim, diagramYDim
  );

	// Sanity checks
	checkCuda(cudaDeviceSynchronize());
	//getLastCudaError("Kernel launch failed!");

	// Copy back
	checkCuda(cudaMemcpy2D(
		hostPixelCanvas, // Dest data pointer
		diagramXDim * pixelChannels * sizeof(unsigned char), // Dest mem alignment
		devicePixelCanvas, // Source data pointer
		diagramXDim * pixelChannels * sizeof(unsigned char), // Source mem alignment
		diagramXDim * pixelChannels * sizeof(unsigned char), // Copy span width (bytes)
		diagramYDim, // Copy span height (elements)
		cudaMemcpyDeviceToHost
	));

	// Cleanup
	checkCuda(cudaFree(devicePixelCanvas));
	checkCuda(cudaFree(deviceNumericCanvas));
	//checkCuda(cudaFreeArray(deviceInputCanvas));
	//checkCuda(cudaDestroyTextureObject(texture));
	checkCuda(cudaDeviceReset());
}
