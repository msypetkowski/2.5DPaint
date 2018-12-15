#pragma once

#include <memory>

#include <QImage>

#include "../brush_settings.h"
#include "../brush_type.h"

class Painter {
public:
	virtual ~Painter() {}

	void setBrush(const BrushSettings& settings) { this->brushSettings = settings; }
	int getWidth() { return w; }
	int getHeight() { return h; }

	void setTexture(QString type, QString path);

	virtual void setDimensions(int w, int h) = 0;
	virtual void *getBufferPtr() = 0;
	virtual void paint(int x, int y) = 0;
	virtual void setBrushType(BrushType type) = 0;

	static std::unique_ptr<Painter> make_painter(bool is_gpu);
protected:
	BrushSettings brushSettings;

	QImage color_image = QImage();
	QImage height_image = QImage();

	int w, h;
};