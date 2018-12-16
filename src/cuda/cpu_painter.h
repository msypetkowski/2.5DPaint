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

	void setDimensions(int w, int h, uchar4 *pbo) override;
	void setBrushType(BrushType type) override;

	void updateWholeDisplay();
	void brushBasic(int mx, int my);
	void brushTextured(int mx, int my);

	void updateBuffer(uchar4 *pbo);
private:
	int getBufferIndex(int x, int y);
	bool inBounds(int x, int y);
	qreal sampleHeight(int x, int y);
	QVector3D getNormal(int x, int y);

	void updateDisplay(int x, int y);
	void updatePainted(int mx, int my);

	void doPainting(int x, int y, uchar4 *pbo) override;

	// buffer for display information
	QVector<uchar4> buffer;

	// internal representation buffers
	QVector<Color> bufferColor;
	QVector<qreal> bufferHeight;

	std::function<void(int, int)> paint_function;
};
