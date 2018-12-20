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
	void setTexture(const std::string& type, const unsigned char *data, int width, int height, int bytes_per_pixel);

    // brush functions
    void brushBasic(int mx, int my);
    void brushTextured(int mx, int my);

private:
	void doPainting(int x, int y, uchar4 *pbo) override;

	//int w, h;

    int color_image_width, color_image_height, height_image_height, height_image_width;
    unsigned char* d_color_texture;
    unsigned char* d_height_texture;
	int color_image_bytes_per_pixel;
	int height_image_bytes_per_pixel;

};