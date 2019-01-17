#include <iostream>
#include <algorithm>
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
    cout << "Average error: " << (float)sum / buf_size << endl;
    cout << "Max error in pixel space (int from [0,255]): " << maxError << endl;

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
    bs.heightPressure = 1.00;
    bs.pressure = 0.5;
    bs.size = 200;
    bs.normalBending = 1;

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
    const auto line_samples = 200;
    const auto dots_samples = 300;
    std::vector<double> gpu_paint_times;
    std::vector<double> cpu_paint_times;
    for (int i=0 ; i < line_samples; ++i) {
        int x = i*3;
        int y = i*5;
        cpu.paint(x, y, pbo1);
        gpu.paint(x, y, pbo2);
        cpu_paint_times.push_back(cpu.getLastPaintingTime());
        gpu_paint_times.push_back(gpu.getLastPaintingTime());
    }

    // random dots
    srand(123);
    for (int i=0 ; i < dots_samples; ++i) {
        int x = rand()%dim1;
        int y = rand()%dim2;
        cpu.paint(x, y, pbo1);
        gpu.paint(x, y, pbo2);
        cpu_paint_times.push_back(cpu.getLastPaintingTime());
        gpu_paint_times.push_back(gpu.getLastPaintingTime());
    }

    cout << "Average time for CPU: " << std::accumulate(cpu_paint_times.begin(), cpu_paint_times.end(), 0.f) / cpu_paint_times.size() << "ms\n";
    cout << "Min time for CPU: " << *std::min_element(cpu_paint_times.begin(), cpu_paint_times.end()) << "ms\n";
    cout << "Average time for GPU: " << std::accumulate(gpu_paint_times.begin(), gpu_paint_times.end(), 0.f) / gpu_paint_times.size() << "ms\n";
    cout << "Min time for GPU: " << *std::min_element(gpu_paint_times.begin(), gpu_paint_times.end()) << "ms\n";
    return compareBuffers(buf_size, pbo1, pbo2);
}


int runTests() {
    cout << "Default brush tests:" << endl;
    brushTest(BrushType::Default);

    cout << endl;

    cout << "Textured brush tests:" << endl;
    brushTest(BrushType::Textured);

    return 0;
}
