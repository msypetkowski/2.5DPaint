#pragma once

#include "cuda_runtime.h"


bool __host__ __device__ in_bounds(int x, int y);
bool __host__ __device__ inBounds(float x, float y);

float __host__ __device__ cosine_fallof(float val, float falloff);

//float4 __host__ __device__ lerp(float4 c1, float4 c2, float4 weight);

float __host__ __device__ normal_from_delta(float dx);

float4 __host__ __device__ make_float4(uchar4 v);
float3 __host__ __device__ make_float3(uchar4 v);

float3 __host__ __device__ interpolate_color(float3 oldColor, float strength, const float3& newColor);