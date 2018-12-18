#include "utils.h"
#include "helper_math.h"
#include "assert.h"

float3 __host__ __device__ interpolate_color(float3 oldColor, float strength, const float3& newColor) {
    float3 ret;
    ret = lerp(oldColor, newColor, strength);
    ret = clamp(ret, make_float3(0,0,0), make_float3(255,255,255));
    return ret;
}

float __host__ __device__ cosine_fallof(float val, float falloff) {
    assert(val >= 0.0f);
    assert(val <= 1.0f);
    val = powf(val, falloff);
    return (cosf(val  * M_PI) + 1.0f) * 0.5f;
}

float __host__ __device__ normal_from_delta(float dx) {
    return dx / sqrtf(dx * dx + 1);
}