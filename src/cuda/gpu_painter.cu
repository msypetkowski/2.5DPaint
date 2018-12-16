#include "gpu_painter.h"

#include <iostream>

#include "helper_cuda.h"

__global__ void writeImageToPBO(uchar4* pbo, int width, int height, int mx, int my);

void GPUPainter::setDimensions(int w, int h, uchar4 *pbo) {
	this->w = w;
	this->h = h;
}

void GPUPainter::setBrushType(BrushType type) {
	//TODO
}

void GPUPainter::doPainting(int x, int y, uchar4 *pbo) {
	const int blockSideLength = 32;
	const dim3 blockSize(blockSideLength, blockSideLength);
	const dim3 blocksPerGrid(
		(w + blockSize.x - 1) / blockSize.x,
		(h + blockSize.y - 1) / blockSize.y);

	writeImageToPBO <<< blocksPerGrid, blockSize >>> (pbo, w, h, x, y);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}

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
