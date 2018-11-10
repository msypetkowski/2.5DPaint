#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_cuda.h"
#include "stdio.h"

#include "kernel.h"

uchar4 *pbo_dptr = NULL;

void setupCuda() {
	checkCudaErrors(cudaSetDevice(0));
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__
void writeImageToPBO(uchar4* pbo, int width, int height, int mx, int my) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	int r2 = (x - mx) * (x - mx) + (y - my) * (y - my);
	
	if (x < width && y < height && mx > -1 && my > -1 && r2 < 25) {
		int index = width - x + (y * width);
		
		pbo[index].w = 0.0f;
		pbo[index].x = 255.0f;
		pbo[index].y = 0.0f;
		pbo[index].z = 255.0f;
	}
}

int cuda_main(int w, int h, int mx, int my) {
	const int blockSideLength = 32;
	const dim3 blockSize(blockSideLength, blockSideLength);
	const dim3 blocksPerGrid(
		(w + blockSize.x - 1) / blockSize.x,
		(h + blockSize.y - 1) / blockSize.y);

	if (pbo_dptr) {
		writeImageToPBO << <blocksPerGrid, blockSize >> > (pbo_dptr, w, h, mx, my);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
	} else {
		printf("Failed to map pbo pointer (pbo_dptr = NULL)\n");
	}
	return cudaGetLastError();
}