#pragma once

#include <string>

#include "painter.h"

#include "cuda_runtime.h"

void setupCuda();

class GPUPainter : public Painter {
public:
	~GPUPainter() override {}

	void setDimensions(int w, int h, uchar4 *pbo) override;
	void setBrushType(BrushType type) override;
	int getBufferIndex(int x, int y) override;
	void setTexture(const std::string& type, const unsigned char *data, int width, int height);

private:
	void doPainting(int x, int y, uchar4 *pbo) override;

	int w, h;
	int image_width, image_height;
	unsigned char* d_color_texture;
	unsigned char* d_height_texture;
	uchar4* d_buffer_pbo = nullptr;

	// internal representation buffers
	float3* d_buffer_color = nullptr;
	float* d_buffer_height = nullptr;

};