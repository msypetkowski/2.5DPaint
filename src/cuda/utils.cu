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

int2 get_coords(int x, int y, int w, int h, int width, int height) {
    const auto pixel_x = int(x / float(w) * width);
    const auto pixel_y = int(y / float(w) * height);
    return make_int2(pixel_x, pixel_y);
}