#pragma once

#include <string>

#include "painter.h"

#include "cuda_runtime.h"
#include "../kernel_params.h"


class GPUPainter : public Painter {
public:
	~GPUPainter() override {}

	void setDimensions(int w, int h, uchar4 *pbo) override;
	void updateWholeDisplay();
	void setBrushType(BrushType type) override;
	void setTexture(const std::string& type, const unsigned char *data, int width, int height, int bytes_per_pixel);

	KernelArgs& getKernelArgs() {return args;}

    // brush functions
    void brushBasic(int mx, int my);
    void brushTextured(int mx, int my);

private:
	void doPainting(int x, int y, uchar4 *pbo) override;
    void clearImage(float3 color, float height) override;

	KernelArgs args;
};