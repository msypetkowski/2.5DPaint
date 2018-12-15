#pragma once
#include "cuda_runtime.h"

#include <functional>
#include <QVector>
#include <QColor>
#include <QImage>

#include "../brush_settings.h"
#include "../brush_type.h"
#include "painter.h"

class CPUPainter : public Painter {
public:
	CPUPainter() {}
	~CPUPainter() override {}

	void setDimensions(int w, int h) override;

	void updateWholeDisplay();
	void brushBasic(int mx, int my);
	void brushTextured(int mx, int my);

	void* getBufferPtr() override { return &buffer[0]; }

	void setBrushType(BrushType type) override;

	void paint(int x, int y) override;
private:
	int getBufferIndex(int x, int y);
	bool inBounds(int x, int y);
	qreal sampleHeight(int x, int y);
	QVector3D getNormal(int x, int y);

	void updateDisplay(int x, int y);
	void updatePainted(int mx, int my);

	// buffer for display information
	QVector<uchar4> buffer;

	// internal representation buffers
	QVector<Color> bufferColor;
	QVector<qreal> bufferHeight;

	std::function<void(int, int)> paint_function;
};
