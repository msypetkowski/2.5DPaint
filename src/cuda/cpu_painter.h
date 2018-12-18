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

	void setTexture(const QString& type, const QString& path);
	const QImage& getTexture(const QString& type) const;

	void updateBuffer(uchar4 *pbo);
private:
	int getBufferIndex(int x, int y) override;

	void updatePainted(int mx, int my);

	void doPainting(int x, int y, uchar4 *pbo) override;

	// textures
	QImage color_image = QImage();
	QImage height_image = QImage();

	// internal representation buffers
	QVector<float3> bufferColor;
	QVector<float> bufferHeight;

	// buffer for display information
	QVector<uchar4> buffer;

	std::function<void(int, int)> paint_function;
};
