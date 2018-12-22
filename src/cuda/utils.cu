#include "utils.h"
#include "helper_math.h"


float3 interpolate_color(float3 oldColor, float strength, const float3& newColor) {
    float3 ret;
    ret = lerp(oldColor, newColor, strength);
    ret = clamp(ret, make_float3(0,0,0), make_float3(255,255,255));
    return ret;
}

float cosine_fallof(float val, float falloff) {
    val = powf(val, falloff);
    return (cosf(val  * (float)M_PI) + 1.0f) * 0.5f;
}


float normal_from_delta(float dx) {
    return dx / sqrtf(dx * dx + 1);
}