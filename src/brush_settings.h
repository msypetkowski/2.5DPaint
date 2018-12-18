#pragma once

#include "cuda_runtime.h"

#include "color.h"

struct BrushSettings {
	double pressure;
	double heightPressure;
	double size;
	double falloff;
	float3 color;
};
