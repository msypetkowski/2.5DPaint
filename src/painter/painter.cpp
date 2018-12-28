#include "painter.h"

#include  "cpu_painter.h"
#include  "gpu_painter.h"


std::unique_ptr<Painter> Painter::make_painter(bool is_gpu) {
    if (is_gpu) {
        return std::make_unique<GPUPainter>();
    } else {
        return std::make_unique<CPUPainter>();
    }
}

void Painter::paint(int x, int y, uchar4 *pbo) {
    doPainting(x, y, pbo);
}

void Painter::clear() {
    clearImage(backgroundColor, 0.0f);
}
