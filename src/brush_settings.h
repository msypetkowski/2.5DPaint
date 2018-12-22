#pragma once

#include "cuda_runtime.h"


struct BrushSettings {
	float pressure;
	float heightPressure;
	float size;
	float falloff;
	float3 color;
};
