#pragma once

#include "painter.h"

#include "cuda_runtime.h"

void setupCuda();

class GPUPainter : public Painter {
public:
	~GPUPainter() override {}

	void setDimensions(int w, int h) override;
	void setBrushType(BrushType type) override;

private:
	void doPainting(int x, int y, uchar4 *pbo) override;

	int w, h;
};