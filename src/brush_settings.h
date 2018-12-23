#pragma once

#include "cuda_runtime.h"


struct BrushSettings {
	float pressure;
	float heightPressure;
	float size;
	float falloff;
	float3 color;

	bool renderNormals = false;
	float normalBending = 100.0f; // normal bending parameter (see getNormal() function)
};
