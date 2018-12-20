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
	void setTexture(const std::string& type, const unsigned char *data, int width, int height);

    // brush functions
    void brushBasic(int mx, int my);
    void brushTextured(int mx, int my);

private:
	void doPainting(int x, int y, uchar4 *pbo) override;

    int image_width, image_height;
    unsigned char* d_color_texture;
    unsigned char* d_height_texture;

};