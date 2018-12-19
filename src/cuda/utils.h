#pragma once

#include "cuda_runtime.h"

__host__ __device__
float cosine_fallof(float val, float falloff);

__host__ __device__
float normal_from_delta(float dx);

__host__ __device__
float3 interpolate_color(float3 oldColor, float strength, const float3& newColor);