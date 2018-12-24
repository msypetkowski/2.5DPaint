#include <iostream>
using namespace std;

#include "cuda_runtime.h"
#include "helper_cuda.h"

#include "cpu_painter.h"
#include "gpu_painter.h"

float compareBuffers(int buf_size, uchar4 *buf1, uchar4 *buf2) {
	uchar4 *b1 = new uchar4[buf_size];
	uchar4 *b2 = new uchar4[buf_size];
	checkCudaErrors(cudaMemcpy(b1, buf1, buf_size * sizeof(uchar4), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(b2, buf2, buf_size * sizeof(uchar4), cudaMemcpyDeviceToHost));

	int sum=0;
	int maxError=-1;
	for (int i = 0 ; i < buf_size ; ++i) {
		sum += abs(b1[i].x - b2[i].x);
		sum += abs(b1[i].y - b2[i].y);
		sum += abs(b1[i].z - b2[i].z);
		sum += abs(b1[i].w - b2[i].w);
		maxError = max(maxError, abs(b1[i].x - b2[i].x));
		maxError = max(maxError, abs(b1[i].y - b2[i].y));
		maxError = max(maxError, abs(b1[i].z - b2[i].z));
		maxError = max(maxError, abs(b1[i].w - b2[i].w));
	}
	cout << "Average error:" << (float)sum / buf_size << endl;
	cout << "Max error:" << maxError << endl;

	return maxError;
}

float brushTest(BrushType brushType) {
	int dim1, dim2;
	dim1 = 1234;
	dim2 = 1234;
	int buf_size = dim1 * dim2;
	uchar4 *pbo1;
	checkCudaErrors(cudaMalloc((void **) &pbo1, buf_size * sizeof(uchar4)));
	uchar4 *pbo2;
	checkCudaErrors(cudaMalloc((void **) &pbo2, buf_size * sizeof(uchar4)));

	BrushSettings bs;
	bs.color = float3({123,20,220});
	bs.falloff = 0.5;
	bs.heightPressure = 0.02;
	bs.pressure = 0.5;
	bs.size = 83;
	bs.normalBending = 100;

	CPUPainter cpu;
	cpu.setDimensions(dim1, dim2, pbo1);
	cpu.clear();
	cpu.updateWholeDisplay();
	cpu.sendBufferToDevice(pbo1);
	cpu.setBrushType(brushType);
	cpu.setBrush(bs);

	GPUPainter gpu;
	gpu.setDimensions(dim1, dim2, pbo2);
	gpu.clear();
	gpu.updateWholeDisplay();
	gpu.setBrushType(brushType);
	gpu.setBrush(bs);

	if (brushType == BrushType::Textured) {
		cpu.setTexture("colorFilename", "textures/RockColor.png");
		cpu.setTexture("heightFilename", "textures/RocksDistortion.png");

		QImage tmp = cpu.getTexture("colorFilename").convertToFormat(QImage::Format_RGB888);
		gpu.setTexture("colorFilename", tmp.bits(), tmp.width(),tmp.height(), tmp.bytesPerLine()/tmp.width());

		tmp = cpu.getTexture("heightFilename").convertToFormat(QImage::Format_RGB888);
		gpu.setTexture("heightFilename", tmp.bits(), tmp.width(),tmp.height(), tmp.bytesPerLine()/tmp.width());
	}

	// straight line
	for (int i=0 ; i < 200 ; ++i) {
		int x = i*3;
		int y = i*5;
		cpu.paint(x, y, pbo1);
		gpu.paint(x, y, pbo2);
	}

	// random dots
	srand(123);
	for (int i=0 ; i < 300 ; ++i) {
		int x = rand()%dim1;
		int y = rand()%dim2;
		cpu.paint(x, y, pbo1);
		gpu.paint(x, y, pbo2);
	}

	return compareBuffers(buf_size, pbo1, pbo2);
}


int runTests() {
	float e1 = brushTest(BrushType::Default);
	float e2 = brushTest(BrushType::Textured);
	cout << "Max errors in color space (0-255): " << endl;
	cout << "Max error default: " << e1 << endl;
	cout << "Max error textured: " << e2 << endl;
	return 0;
}
