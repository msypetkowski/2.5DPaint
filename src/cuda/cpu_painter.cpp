#include "cpu_painter.h"

#include <iostream>

#include <QtMath>
#include <QVector3D>
#include <QElapsedTimer>
#include <helper_math.h>

#include "helper_cuda.h"
#include "utils.h"


std::pair<int, int> get_coords(const QImage &image, int x, int y, int w, int h) {
    const auto pixel_x = int(x / float(w) * image.width());
    const auto pixel_y = int(y / float(w) * image.height());
    return {pixel_x, pixel_y};
}

void CPUPainter::updateWholeDisplay() {
    for (int x = 0; x < w; ++x) {
        for (int y = 0; y < h; ++y) {
            updateDisplayPixel(x, y);
        }
    }
}

void CPUPainter::setDimensions(int w1, int h1, uchar4 *pbo) {
    w = w1;
    h = h1;

    int buf_size = w * h;

    printf("init/resize cpu buffers (%d, %d)\n", w, h);
    buffer.resize(buf_size);
    bufferColor.resize(buf_size);
    bufferHeight.resize(buf_size);

    checkCudaErrors(cudaMemcpy(buffer.data(), pbo, buf_size * sizeof(uchar4), cudaMemcpyDeviceToHost));

    float3 fill = make_float3(75);
    bufferColor.fill(fill);
    bufferHeight.fill(0.0);
    // cpu_buffer.fill(fill);

    // assign pointers
    buffer_pbo = buffer.data();
    buffer_color = bufferColor.data();
    buffer_height = bufferHeight.data();

    updateWholeDisplay();
}

void CPUPainter::brushBasic(int mx, int my) {
    int maxRadius = brushSettings.size / 2;
    for (int x = mx - maxRadius + 1; x < mx + maxRadius; ++x) {
        for (int y = my - maxRadius + 1; y < my + maxRadius; ++y) {
            brushBasicPixel(x, y, mx, my);
        }
    }
    updatePainted(mx, my);
}

void CPUPainter::updatePainted(int mx, int my) {
    qreal maxRadius = brushSettings.size / 2;
    for (int x = mx - maxRadius + 1; x < mx + maxRadius; ++x) {
        for (int y = my - maxRadius + 1; y < my + maxRadius; ++y) {
            if (!in_bounds(x, y))
                continue;
            updateDisplayPixel(x, y);
        }
    }
}

void CPUPainter::brushTextured(int mx, int my) {
    qreal maxRadius = brushSettings.size / 2;
    if (color_image.isNull() || height_image.isNull()) {
        std::clog << "No texture set\n";
        return;
    }
    for (int x = mx - maxRadius + 1; x < mx + maxRadius; ++x) {
        for (int y = my - maxRadius + 1; y < my + maxRadius; ++y) {
            if (!in_bounds(x, y))
                continue;
            qreal radius = qSqrt((x - mx) * (x - mx) + (y - my) * (y - my));
            if (radius > maxRadius) {
                continue;
            }
            int i = getBufferIndex(x, y);

            qreal strength = brushSettings.pressure * cosine_fallof(radius / maxRadius, brushSettings.falloff);
            const auto color_coords =
                    get_coords(color_image, x - mx + maxRadius, y - my + maxRadius, maxRadius * 2, maxRadius * 2);
            const auto pixel = color_image.pixel(color_coords.first, color_coords.second);
            buffer_color[i] = interpolate_color(
                    buffer_color[i],
                    strength,
                    make_float3(qRed(pixel), qGreen(pixel), qBlue(pixel)));

            const auto height_coords =
                    get_coords(height_image, x - mx + maxRadius, y - my + maxRadius, maxRadius * 2, maxRadius * 2);
            const auto height = qRed(height_image.pixel(height_coords.first, height_coords.second)) * 0.001f;
            strength = brushSettings.heightPressure * height * cosine_fallof(radius / maxRadius, brushSettings.falloff);
            bufferHeight[i] = qBound(-1.0, bufferHeight[i] + strength, 1.0);
        }
    }
    updatePainted(mx, my);
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
        case BrushType::Third:
            std::clog << "Warning: chose unused brush\n";
            break;
        default:
            throw std::runtime_error("Invalid brush type: "
                                     + std::to_string(static_cast<int>(type)));
    }
}

void CPUPainter::doPainting(int x, int y, uchar4 *pbo) {
    QElapsedTimer performanceTimer;
    const auto buf_size = w * h * sizeof(uchar4);

    //TODO: leaving it here, this logic used to be in previewGLWidget, although I don't know why
    //performanceTimer.restart();
    //checkCudaErrors(cudaMemcpy(&buffer[0], pbo, buf_size, cudaMemcpyDeviceToHost));
    //const auto memcpy_d2h = performanceTimer.nsecsElapsed();

    performanceTimer.restart();
    paint_function(x, y);
    const auto painting_duration = performanceTimer.nsecsElapsed();

    performanceTimer.restart();
    checkCudaErrors(cudaMemcpy(pbo, &buffer[0], buf_size, cudaMemcpyHostToDevice));
    const auto memcpy_h2d = performanceTimer.nsecsElapsed();

    std::clog << "Painting: " << painting_duration / 1e6f << "ms\n";
    std::clog << "Copying: " << memcpy_h2d / 1e6f << "ms\n";
}

void CPUPainter::updateBuffer(uchar4 *pbo) {
    if (w <= 0 || h <= 0 || buffer.size() == 0)
        return;
    const auto buf_size = w * h * sizeof(uchar4);
    checkCudaErrors(cudaMemcpy(buffer.data(), pbo, buf_size, cudaMemcpyDeviceToHost));
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

