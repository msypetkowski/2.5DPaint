#include "gpu_painter.h"

#include <iostream>

#include "helper_cuda.h"
#include "utils.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "stdio.h"

__global__ void writeImageToPBO(uchar4* pbo, int width, int height, int mx, int my);

void GPUPainter::setDimensions(int w, int h, uchar4 *pbo) {
	this->w = w;
	this->h = h;

	int buf_size = w * h;

	printf("init/resize gpu buffers (%d, %d)\n", w, h);

	cudaFree(d_buffer_color);
	cudaFree(d_buffer_height);
	cudaFree(d_buffer_pbo);

	checkCudaErrors(cudaMalloc((void**)&d_buffer_pbo, buf_size * sizeof(uchar4)));
	//cudaMemset(buffer_pbo, 0, buf_size* sizeof(uchar4));

	checkCudaErrors(cudaMalloc((void**)&d_buffer_height, buf_size * sizeof(float)));
	//cudaMemset(buffer_height, 0, buf_size * sizeof(float));

	checkCudaErrors(cudaMalloc((void**)&d_buffer_color, buf_size * sizeof(float3)));
	//cudaMemset(buffer_color, 0, buf_size * sizeof(float3));

	//updateWholeDisplay();
}

int GPUPainter::getBufferIndex(int x, int y) {
    return w - 1 - x + (y * w);
}

void GPUPainter::setBrushType(BrushType type) {
	/*switch (type) {
	case BrushType::Default:
		paint_function = std::bind(&CPUPainter::brushBasic, this, _1, _2);
		break;
	case BrushType::Textured:
		paint_function = std::bind(&CPUPainter::brushTextured, this, _1, _2);
		break;
	case BrushType::Third:
		std::clog << "Warning: chose unused brush\n";
		break;
	default:
		throw std::runtime_error("Invalid brush type: "
			+ std::to_string(static_cast<int>(type)));
	}*/
}

void GPUPainter::setTexture(const std::string& type, const unsigned char *data, int width, int height) {

	image_height = height;
	image_width = width;

	if (type == "colorFilename") {
		checkCudaErrors(cudaMalloc((void**)&d_color_texture, sizeof(data)));
		checkCudaErrors(cudaMemcpy(d_color_texture, data, sizeof(data), cudaMemcpyHostToDevice));
	}
	else {
		checkCudaErrors(cudaMalloc((void**)&d_height_texture, sizeof(data)));
		checkCudaErrors(cudaMemcpy(d_height_texture, data, sizeof(data), cudaMemcpyHostToDevice));
	}
	

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
	(cudaSetDevice(0));
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
