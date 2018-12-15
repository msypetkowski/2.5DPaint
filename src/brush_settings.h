#pragma once

#include "cuda_runtime.h"

#include "color.h"

struct BrushSettings {
	qreal pressure;
	qreal heightPressure;
	qreal size;
	qreal falloff;
	Color color;
};
