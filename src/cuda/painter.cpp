#include "painter.h"

#include  "cpu_painter.h"

std::unique_ptr<Painter> Painter::make_painter(bool is_gpu) {
	if (is_gpu) {
		return nullptr;
	} else {
		return std::make_unique<CPUPainter>();
	}
}

void Painter::setTexture(QString type, QString path) {
	if (type == "colorFilename") {
		color_image = QImage(path);
	} else {
		height_image = QImage(path);
	}
}
