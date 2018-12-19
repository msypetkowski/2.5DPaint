#pragma once

#include <memory>
#include <functional>

#include "../brush_settings.h"
#include "../brush_type.h"

class Painter {
public:
	virtual ~Painter() {}

	void setBrush(const BrushSettings& settings) { this->brushSettings = settings; }
	int getWidth() { return w; }
	int getHeight() { return h; }

	void paint(int x, int y, uchar4 *pbo);

	virtual void setDimensions(int w, int h, uchar4 *pbo) = 0;
	virtual void setBrushType(BrushType type) = 0;
    int getBufferIndex(int x, int y);

	void brushBasicPixel(int x, int y, int mx, int my);
	void updateDisplayPixel(int x, int y);
	float3 getNormal(int x, int y);
	float sampleHeight(int x, int y);

	bool in_bounds(int x, int y);

	static std::unique_ptr<Painter> make_painter(bool is_gpu);
private:
	virtual void doPainting(int x, int y, uchar4 *pbo) = 0;

protected:
	BrushSettings brushSettings;

	int w, h;

	// display buffer
	uchar4* buffer_pbo = nullptr;

	// internal representation buffers
	float3* buffer_color = nullptr;
	float* buffer_height = nullptr;

	//paint function
    std::function<void(int, int)> paint_function;
};