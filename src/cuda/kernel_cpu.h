#pragma once
#include "cuda_runtime.h"

#include <functional>
#include <QVector>
#include <QColor>
#include <QImage>

#include "../brush.h"



class CPUPainter {
public:
	CPUPainter() {}

	void setDimensions(int w, int h);
	void setBrush(const BrushSettings& bs) { brushSettings = bs; }

	void updateWholeDisplay();
	void brushBasic(int mx, int my);
	void brushTextured(int mx, int my);

	int getWidth() { return w; }
	int getHeight() { return h; }

	void* getBufferPtr() { return &cpuBuffer[0]; }

	void setTexture(QString type, QString path);

	std::function<void(int, int)> paint;
private:
	int getBufferIndex(int x, int y);
	bool inBounds(int x, int y);
	qreal sampleHeight(int x, int y);
	QVector3D getNormal(int x, int y);

	void updateDisplay(int x, int y);
	void updatePainted(int mx, int my);

	// buffer for display information
	QVector<uchar4> cpuBuffer;

	// internal representation buffers
	QVector<Color> cpuBufferColor;
	QVector<qreal> cpuBufferHeight;

	BrushSettings brushSettings;
	QImage color_image = QImage();
	QImage height_image = QImage();
	int w, h;
};
