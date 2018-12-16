#include "painter.h"

#include <iostream>

#include <QOpenGLBuffer>

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
    QElapsedTimer performanceTimer;
	performanceTimer.restart();
	doPainting(x, y, pbo);
	const auto elapsed_time = performanceTimer.nsecsElapsed();
	std::clog << "Time netto: " << elapsed_time/1e6f << "ms\n";
}
