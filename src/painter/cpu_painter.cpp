#include "cpu_painter.h"

#include <iostream>
#include <chrono>

#include <QtMath>
#include <QVector3D>
#include <helper_math.h>

#include "helper_cuda.h"
#include "utils.h"


bool CPUPainter::inBounds(int x, int y) {
    return x >= 0 && x < w && y >= 0 && y < h;
}

float CPUPainter::sampleHeight(int x, int y) {
    x = clamp(x, 0, w - 1);
    y = clamp(y, 0, h - 1);
    return buffer_height[getBufferIndex(x, y)];
}

float3 CPUPainter::getNormal(int x, int y) {
    float dx = 0.0f, dy = 0.0f;

    auto mid = sampleHeight(x, y);
    auto left = sampleHeight(x - 1, y);
    auto right = sampleHeight(x + 1, y);
    auto top = sampleHeight(x, y + 1);
    auto bottom = sampleHeight(x, y - 1);

    dx += normal_from_delta(mid - right) / 2;
    dx -= normal_from_delta(mid - left) / 2;

    dy += normal_from_delta(mid - top) / 2;
    dy -= normal_from_delta(mid - bottom) / 2;

    dx *= brushSettings.normalBending;
    dy *= brushSettings.normalBending;

    dx = dx / sqrtf(dx * dx + dy * dy + 1);
    dy = dy / sqrtf(dx * dx + dy * dy + 1);

    auto ret = make_float3(dx, dy, sqrtf(fabsf(1.0f - dx * dx - dy * dy)));
    return normalize(ret);
}

int CPUPainter::getBufferIndex(int x, int y) {
    return w - 1 - x + (y * w);
}

void CPUPainter::setDimensions(int w1, int h1, uchar4 *pbo) {
    w = w1;
    h = h1;

    int buf_size = w * h;

    printf("[CPU] init/resize cpu buffers (%d, %d)\n", w, h);
    buffer.resize(buf_size);
    bufferColor.resize(buf_size);
    swapbufferColor.resize(buf_size);
    bufferHeight.resize(buf_size);
    swapbufferHeight.resize(buf_size);
    maskBuffer.resize(buf_size);

    checkCudaErrors(cudaMemcpy(buffer.data(), pbo, buf_size * sizeof(uchar4), cudaMemcpyDeviceToHost));

    Painter::clear();

    // assign pointers
    buffer_pbo = buffer.data();
    buffer_color = bufferColor.data();
    buffer_height = bufferHeight.data();
    swap_buffer_height = swapbufferHeight.data();
    swap_buffer_color = swapbufferColor.data();

    updateWholeDisplay();
}

void CPUPainter::clearImage(float3 color, float height) {

    bufferColor.fill(color);
    swapbufferColor.fill(color);
    bufferHeight.fill(height);
    swapbufferHeight.fill(height);
    // no need to fill mask buffer

    std::clog << "[CPU] Clear image\n";
}


void CPUPainter::setBrushType(BrushType type) {
    using namespace std::placeholders;
    switch (type) {
        case BrushType::Default:
            paint_function = std::bind(&CPUPainter::brushBasic, this, _1, _2);
            break;
        case BrushType::Textured:
            paint_function = std::bind(&CPUPainter::brushTextured, this, _1, _2);
            break;
        case BrushType::Smooth:
            paint_function = std::bind(&CPUPainter::brushSmooth, this, _1, _2);
            break;
        case BrushType::Inflate:
            paint_function = std::bind(&CPUPainter::brushInflate, this, _1, _2);
            break;
        default:
            throw std::runtime_error("Invalid brush type: "
                                     + std::to_string(static_cast<int>(type)));
    }
}

void CPUPainter::doPainting(int x, int y, uchar4 *pbo) {
    const auto buf_size = w * h * sizeof(uchar4);

    auto start_time = std::chrono::steady_clock::now();
    paint_function(x, y);
    last_painting_time = (float)std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start_time).count() / (float)1e6;

    start_time = std::chrono::steady_clock::now();
    checkCudaErrors(cudaMemcpy(pbo, &buffer[0], buf_size, cudaMemcpyHostToDevice));
    const auto memcpy_h2d = (float)std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start_time).count() / (float)1e6;

    std::clog << "[CPU] Painting time: " << last_painting_time << " ms\n";
    std::clog << "[CPU] Copying pbo (host -> device): " << memcpy_h2d << " ms\n";
    std::clog << "[CPU] Sum: " << (memcpy_h2d + last_painting_time) << " ms\n";
}

void CPUPainter::pullBufferFromDevice(uchar4 *pbo) {
    if (w <= 0 || h <= 0 || buffer.size() == 0)
        return;
    const auto buf_size = w * h * sizeof(uchar4);
    checkCudaErrors(cudaMemcpy(buffer.data(), pbo, buf_size, cudaMemcpyDeviceToHost));
}

void CPUPainter::sendBufferToDevice(uchar4 *pbo) {
    if (w <= 0 || h <= 0 || buffer.size() == 0)
        return;
    const auto buf_size = w * h * sizeof(uchar4);
    checkCudaErrors(cudaMemcpy(pbo, buffer.data(), buf_size, cudaMemcpyHostToDevice));
}

void CPUPainter::setTexture(const QString &type, const QString &path) {
    if (type == "colorFilename") {
        color_image = QImage(path);
    } else {
        height_image = QImage(path);
    }
}

const QImage &CPUPainter::getTexture(const QString &type) const {
    if (type == "colorFilename") {
        return color_image;
    } else {
        return height_image;
    }
}

void CPUPainter::swapHeightBuffer() {
    float* tmp = buffer_height;
    buffer_height = swap_buffer_height;
    swap_buffer_height = tmp;
}

void CPUPainter::swapColorBuffer() {
    float3* tmp = buffer_color;
    buffer_color = swap_buffer_color;
    swap_buffer_color = tmp;
}


/**********************************************************************************************************************/
/*
 * Brush functions
 */

void CPUPainter::brushBasic(int mx, int my) {
    float brush_radius = brushSettings.size / 2.0f;
    for (int x = mx - brush_radius + 1; x < mx + brush_radius; ++x) {
        for (int y = my - brush_radius + 1; y < my + brush_radius; ++y) {
            if (!inBounds(x, y))
                continue;
            float radius = sqrtf((x - mx) * (x - mx) + (y - my) * (y - my));
            if (radius > brush_radius) {
                continue;
            }
            int i = getBufferIndex(x, y);

            // paint color
            float strength = brushSettings.pressure * cosine_fallof(radius / brush_radius, brushSettings.falloff);
            float3 color = interpolate_color(buffer_color[i], strength, brushSettings.color);
            buffer_color[i] = color;
            swap_buffer_color[i] = color;

            // paint height
            strength = brushSettings.heightPressure * cosine_fallof(radius / brush_radius, brushSettings.falloff);

            float result = clamp(buffer_height[i] + strength, -1000.0f, 1000.0f);

            buffer_height[i] = result;
            swap_buffer_height[i] = result;
        }
    }
    updatePainted(mx, my);
}

void CPUPainter::brushTextured(int mx, int my) {
    float maxRadius = brushSettings.size / 2;

    if (color_image.isNull() || height_image.isNull()) {
        std::clog << "[CPU] No texture set\n";
        return;
    }

    for (int x = mx - maxRadius + 1; x < mx + maxRadius; ++x) {
        for (int y = my - maxRadius + 1; y < my + maxRadius; ++y) {

            if (!inBounds(x, y))
                continue;

            float radius = sqrtf((x - mx) * (x - mx) + (y - my) * (y - my));

            if (radius > maxRadius) {
                continue;
            }
            int i = getBufferIndex(x, y);

            float strength = brushSettings.pressure * cosine_fallof(radius / maxRadius, brushSettings.falloff);

            const auto color_coords = get_coords(x - mx + maxRadius,
                                                 y - my + maxRadius,
                                                 brushSettings.size,
                                                 brushSettings.size,
                                                 color_image.width(),
                                                 color_image.height());

            const auto pixel = color_image.pixel(color_coords.x, color_coords.y);

            const auto color = interpolate_color(buffer_color[i],
                                                strength,
                                                make_float3(qRed(pixel), qGreen(pixel), qBlue(pixel)));
            buffer_color[i] = color;
            swap_buffer_color[i] = color;

            const auto height_coords = get_coords(x - mx + maxRadius,
                                                  y - my + maxRadius,
                                                  brushSettings.size,
                                                  brushSettings.size,
                                                  height_image.width(),
                                                  height_image.height());

            const auto height = qRed(height_image.pixel(height_coords.x, height_coords.y)) * 0.001f;

            strength = brushSettings.heightPressure * height * cosine_fallof(radius / maxRadius, brushSettings.falloff);

            float result = clamp(buffer_height[i] + strength, -1.0f, 1.0f);

            buffer_height[i] = result;
            swap_buffer_height[i] = result;
        }
    }
    updatePainted(mx, my);
}

void CPUPainter::brushSmooth(int mx, int my) {
    float brush_radius = brushSettings.size / 2.0f;
    for (int x = mx - brush_radius + 1; x < mx + brush_radius; ++x) {
        for (int y = my - brush_radius + 1; y < my + brush_radius; ++y) {
            if (!inBounds(x, y))
                continue;
            float radius = sqrtf((x - mx) * (x - mx) + (y - my) * (y - my));
            if (radius > brush_radius) {
                continue;
            }
            int i = getBufferIndex(x, y);

            float strength = cosine_fallof(radius / brush_radius, brushSettings.falloff);

            // apply convolution filter
            auto mid = sampleHeight(x, y);

            auto left = sampleHeight(x - 1, y);
            auto right = sampleHeight(x + 1, y);

            auto top = sampleHeight(x, y + 1);
            auto bottom = sampleHeight(x, y - 1);

            auto topleft = sampleHeight(x - 1, y - 1);
            auto topright = sampleHeight(x + 1, y - 1);
            auto bottomleft = sampleHeight(x - 1, y + 1);
            auto bottomright = sampleHeight(x + 1, y + 1);

            float result = (mid + left + right + top + bottom + topleft + topright + bottomleft + bottomright) / 9.0f;

            swap_buffer_height[i] = mid + strength * (result - mid);
        }
    }
    swapHeightBuffer();
    updatePainted(mx, my);
}


void CPUPainter::brushInflate(int mx, int my)
{
    float brush_draw_radius = brushSettings.size / 2;
    float brush_radius = brushSettings.size;

    // clear part of mask buffer that will be used
    for (int x = mx - brush_radius + 1; x < mx + brush_radius; ++x) {
        for (int y = my - brush_radius + 1; y < my + brush_radius; ++y) {
            if (!inBounds(x, y))
                continue;
            int i = getBufferIndex(x,y);
            maskBuffer[i] = false;
        }
    }

    float3 origin = {(float)mx, (float)my, buffer_height[getBufferIndex(mx,my)]};
    for (int x = mx - brush_radius + 1; x < mx + brush_radius; ++x) {
        for (int y = my - brush_radius + 1; y < my + brush_radius; ++y) {
            if (!inBounds(x, y))
                continue;
            int i = getBufferIndex(x,y);
            float3 location = {(float)x, (float)y, buffer_height[i]};
            float radius = length(location - origin);
            // if (radius > brush_radius) {
            //     continue;
            // }

            float3 normal = getNormal(x, y);
            // TODO: make smooth fade-out
            float strength = brushSettings.heightPressure * cosine_fallof(fminf(radius / brush_draw_radius, 1.0), brushSettings.falloff);

            // this formula can be easily changed for other similar brushes e.g. flatten, pinch or clay
            // interpolation and faloff settings should be adjusted accordingly
            float3 newLocation = location + normal * strength;

            maskBuffer[i] = maskBuffer[i] || strength < 0.001;
            int newX = (int)(newLocation.x + 0.5), newY = (int)(newLocation.y + 0.5);

            if (!inBounds(newX, newY))
                continue;

            int j = getBufferIndex(newX, newY);
            swap_buffer_color[j] = buffer_color[i];
            swap_buffer_height[j] = newLocation.z;
            maskBuffer[j] = true;
        }
    }

    // interpolate missing values (TODO: improve / get rid of noise)
    for (int x = mx - brush_radius + 1; x < mx + brush_radius; ++x) {
        for (int y = my - brush_radius + 1; y < my + brush_radius; ++y) {
            int i = getBufferIndex(x,y);
            if (!inBounds(x, y) || maskBuffer[i])
                continue;
            float closestHeight = 0;
            float3 closestColor({0,0,255});
            float minDist = std::numeric_limits<float>::infinity();
            int maxPositionOffset = (int)brushSettings.heightPressure + 10;
            for (int x2 = x - maxPositionOffset + 1; x2 < x + maxPositionOffset; ++x2) {
                for (int y2 = y - maxPositionOffset + 1; y2 < y + maxPositionOffset; ++y2) {
                    if (!inBounds(x2, y2))
                        continue;
                    int j = getBufferIndex(x2,y2);
                    float dist = sqrt((x2-x)*(x2-x) + (y2-y)*(y2-y));
                    if (maskBuffer[j] && dist < minDist) {
                        closestHeight = swap_buffer_height[j];
                        closestColor = swap_buffer_color[j];
                        minDist = dist;
                    }
                }
            }
            swap_buffer_color[i] = closestColor;
            swap_buffer_height[i] = closestHeight;
        }
    }

    // keep buffers and swap buffers consistency
    for (int x = mx - brush_radius + 1; x < mx + brush_radius; ++x) {
        for (int y = my - brush_radius + 1; y < my + brush_radius; ++y) {
            if (!inBounds(x, y))
                continue;
            int i = getBufferIndex(x,y);
            buffer_color[i] = swap_buffer_color[i];
            buffer_height[i] = swap_buffer_height[i];
        }
    }

    // updatePainted(mx, my);
    float maxRadius = brush_radius;
    for (int x = mx - maxRadius + 1; x < mx + maxRadius; ++x) {
        for (int y = my - maxRadius + 1; y < my + maxRadius; ++y) {
            updatePixelDisplay(x, y);
        }
    }
}

/*
 * Function used for shading pixels for display, updates only square painted piece of image affected by brush
 */
void CPUPainter::updatePainted(int mx, int my) {
    float maxRadius = brushSettings.size / 2;
    for (int x = mx - maxRadius + 1; x < mx + maxRadius; ++x) {
        for (int y = my - maxRadius + 1; y < my + maxRadius; ++y) {
            updatePixelDisplay(x, y);
        }
    }
}

/*
 * Function shades all pixels on the screen
 */
void CPUPainter::updateWholeDisplay() {
    for (int x = 0; x < w; ++x) {
        for (int y = 0; y < h; ++y) {
            updatePixelDisplay(x, y);
        }
    }
}

/*
 * Function shades single display pixel
 */
void CPUPainter::updatePixelDisplay(int x, int y) {
    if (!inBounds(x, y))
        return;
    int i = getBufferIndex(x, y);

    auto normal = getNormal(x, y);

    float3 color;

    if (!brushSettings.renderNormals) {
        float3 lighting = normalize(lightDirection);

        // TODO: use lighting vector here
        float shadow = fabsf(dot(normal, lighting));
        shadow = clamp(shadow, 0.0f, 1.0f);

        float specular = 1.0f - length(normal - lighting);
        specular = powf(specular, 8.0f);
        specular = clamp(specular, 0.0f, 1.0f);

        color = lerp(buffer_color[i] * shadow, make_float3(255, 255, 255), specular);
    } else {
        // view normals
        color.x = normal.x * 255.0 / 2 + 255.0 / 2;
        color.y = normal.y * 255.0 / 2 + 255.0 / 2;
        color.z = normal.z * 255;
    }

    color = clamp(color, make_float3(0.0f), make_float3(255.0f));
    buffer_pbo[i] = make_uchar4(color.x, color.y, color.z, 0);
}




