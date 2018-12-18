#pragma once

#include "cuda_runtime.h"


float __host__ __device__ cosine_fallof(float val, float falloff);

float __host__ __device__ normal_from_delta(float dx);

float3 __host__ __device__ interpolate_color(float3 oldColor, float strength, const float3& newColor);