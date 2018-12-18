#include "painter.h"

#include <iostream>

#include <QOpenGLBuffer>
#include <QElapsedTimer>
#include <QtMath>
#include <helper_math.h>

#include  "cpu_painter.h"
#include  "gpu_painter.h"
#include "utils.h"

std::unique_ptr<Painter> Painter::make_painter(bool is_gpu) {
    if (is_gpu) {
        return std::make_unique<GPUPainter>();
    } else {
        return std::make_unique<CPUPainter>();
    }
}

void Painter::paint(int x, int y, uchar4 *pbo) {
    QElapsedTimer performanceTimer;
    performanceTimer.restart();
    doPainting(x, y, pbo);
    const auto elapsed_time = performanceTimer.nsecsElapsed();
    std::clog << "Time netto: " << elapsed_time / 1e6f << "ms\n";
}

bool __host__ __device__ Painter::in_bounds(int x, int y) {
    return x >= 0 && x < w && y >= 0 && y < h;
}

void __host__ __device__ Painter::brushBasicPixel(int x, int y, int mx, int my) {
    if (!in_bounds(x, y))
        return;

    float radius = sqrtf((x - mx) * (x - mx) + (y - my) * (y - my));
    float brush_radius = brushSettings.size / 2.0;
    if (radius > brush_radius) {
        return;
    }
    int i = getBufferIndex(x, y);

    // paint color
    float strength = brushSettings.pressure * cosine_fallof(radius / brush_radius, brushSettings.falloff);
    float3 color = interpolate_color(buffer_color[i], strength, brushSettings.color);
    buffer_color[i] = color;

    // paint height
    strength = brushSettings.heightPressure * cosine_fallof(radius / brush_radius, brushSettings.falloff);
    buffer_height[i] = clamp(buffer_height[i] + strength, -1.0f, 1.0f);
}

void __host__ __device__ Painter::updateDisplayPixel(int x, int y) {
    int i = getBufferIndex(x, y);

    auto normal = getNormal(x, y);

    float3 lighting = normalize(make_float3(0.07f, 0.07f, 1.0f));

    // TODO: use lighting vector here
    float shadow = normal.z * 0.80f - normal.x * 0.1f - normal.y * 0.1f + (sampleHeight(x, y)) / 4.0f;
    shadow = clamp(shadow, 0.0f, 1.0f);

    float specular = 1.0f - length(normal - lighting);
    specular = powf(specular, 8.0f);
    specular = clamp(specular, 0.0f, 1.0f);

    float3 color = lerp(buffer_color[i] * shadow, make_float3(255, 255, 255), specular);

    // view normals (TODO: remove or make normals visualization feature)
    /*color.x = normal.x * 255.0 / 2 + 255.0 / 2;
    color.y = normal.y * 255.0 / 2 + 255.0 / 2;
    color.z = normal.z * 255;*/
    //color = clamp(color, make_float3(0.0f), make_float3(255.0f));
    buffer_pbo[i] = make_uchar4(color.x, color.y, color.z, 0);
}

float3 __host__ __device__ Painter::getNormal(int x, int y) {
    float dx = 0, dy = 0;

    auto mid = sampleHeight(x, y);
    auto left = sampleHeight(x - 1, y);
    auto right = sampleHeight(x + 1, y);
    auto top = sampleHeight(x, y + 1);
    auto bottom = sampleHeight(x, y - 1);

    dx += normal_from_delta(mid - right) / 2;
    dx -= normal_from_delta(mid - left) / 2;

    dy += normal_from_delta(mid - top) / 2;
    dy -= normal_from_delta(mid - bottom) / 2;

    // TODO: make parameter or constant
    dx *= 100;
    dy *= 100;

    dx = dx / sqrtf(dx * dx + dy * dy + 1);
    dy = dy / sqrtf(dx * dx + dy * dy + 1);

    assert(fabsf(dx) <= 1);
    assert(fabsf(dy) <= 1);
    auto ret = make_float3(dx, dy, sqrtf(clamp(1.0f - dx * dx - dy * dy, 0.0f, 1.0f)));
    return normalize(ret);


}

float Painter::sampleHeight(int x, int y) {
    x = clamp(x, 0, w - 1);
    y = clamp(y, 0, h - 1);
    return buffer_height[getBufferIndex(x, y)];
}
