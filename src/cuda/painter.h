#pragma once

#include <memory>
#include <functional>

#include "vector_functions.h"
#include "../brush_settings.h"
#include "../brush_type.h"

class Painter {
public:
	virtual ~Painter() {}

	void setBrush(const BrushSettings& settings) { this->brushSettings = settings; }
	int getWidth() { return w; }
	int getHeight() { return h; }

	void paint(int x, int y, uchar4 *pbo);
	void clear();

	virtual void setDimensions(int w, int h, uchar4 *pbo) = 0;
	virtual void setBrushType(BrushType type) = 0;

	static std::unique_ptr<Painter> make_painter(bool is_gpu);
private:
	virtual void doPainting(int x, int y, uchar4 *pbo) = 0;
	virtual void clearImage(float3 color, float height) = 0;

protected:
	BrushSettings brushSettings;

	float3 backgroundColor = make_float3(125.0f, 125.0f, 125.0f);

	int w, h;

	// display buffer
	uchar4* buffer_pbo = nullptr;

	// internal representation buffers
	float3* buffer_color = nullptr;
	float* buffer_height = nullptr;

	//paint function
    std::function<void(int, int)> paint_function;
};