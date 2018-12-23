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

	void setTexture(const QString& type, const QString& path);
	const QImage& getTexture(const QString& type) const;

	void pullBufferFromDevice(uchar4 *pbo);
	void sendBufferToDevice(uchar4 *pbo);
	float* getHeightBuffer() {return buffer_height; }
	float3* getColorBuffer() {return buffer_color; }
	uchar4* getBuffer() {return buffer_pbo; }

	int getBufferIndex(int x, int y);
    float3 getNormal(int x, int y);
    float sampleHeight(int x, int y);

    bool inBounds(int x, int y);

	// brush functions
	void brushBasic(int mx, int my);
	void brushTextured(int mx, int my);

private:

	void updatePainted(int mx, int my);

	void doPainting(int x, int y, uchar4 *pbo) override;
	void clearImage(float3 color, float height) override;

	// textures
	QImage color_image = QImage();
	QImage height_image = QImage();

	// internal representation buffers
	QVector<float3> bufferColor;
	QVector<float> bufferHeight;

	// buffer for display information
	QVector<uchar4> buffer;

};
