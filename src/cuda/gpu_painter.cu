#include "gpu_painter.h"

#include <iostream>

#include "cuda_runtime.h"
#include "helper_cuda.h"
#include "helper_math.h"
#include "device_launch_parameters.h"


void setupCuda() {
    checkCudaErrors(cudaSetDevice(0));
}

bool __device__ in_bounds(int x, int y, int w, int h) {
    return x >= 0 && x < w && y >= 0 && y < h;
}

float3 __device__ d_interpolate_color(float3 oldColor, float strength, const float3 &newColor) {
    float3 ret;
    ret = lerp(oldColor, newColor, strength);
    ret = clamp(ret, make_float3(0, 0, 0), make_float3(255, 255, 255));
    return ret;
}

float __device__ d_cosine_fallof(float val, float falloff) {
    val = powf(val, falloff);
    return (cosf(val * M_PI) + 1.0f) * 0.5f;
}

float __device__ d_normal_from_delta(float dx) {
    return dx / sqrtf(dx * dx + 1);
}

int __device__ d_getBufferIndex(int x, int y, int w) { return (w - 1 - x + (y * w)); };

int __device__ d_getImageByteIndex(int x, int y, int h, int w, int bpp) { return (w * bpp) * y + x * bpp; };


float __device__ d_sampleHeight(int x, int y, int w, int h, float *buffer_height) {
    x = clamp(x, 0, w - 1);
    y = clamp(y, 0, h - 1);
    return buffer_height[d_getBufferIndex(x, y, w)];
}

float3 __device__ d_getNormal(int x, int y, int w, int h, float *buffer_height) {
    float dx = 0.0f, dy = 0.0f;

    auto mid = d_sampleHeight(x, y, w, h, buffer_height);
    auto left = d_sampleHeight(x - 1, y, w, h, buffer_height);
    auto right = d_sampleHeight(x + 1, y, w, h, buffer_height);
    auto top = d_sampleHeight(x, y + 1, w, h, buffer_height);
    auto bottom = d_sampleHeight(x, y - 1, w, h, buffer_height);

    dx += d_normal_from_delta(mid - right) / 2;
    dx -= d_normal_from_delta(mid - left) / 2;

    dy += d_normal_from_delta(mid - top) / 2;
    dy -= d_normal_from_delta(mid - bottom) / 2;

    // TODO: make parameter or constant
    dx *= 100;
    dy *= 100;

    dx = dx / sqrtf(dx * dx + dy * dy + 1);
    dy = dy / sqrtf(dx * dx + dy * dy + 1);

    auto ret = make_float3(dx, dy, sqrtf(clamp(1.0f - dx * dx - dy * dy, 0.0f, 1.0f)));
    return normalize(ret);
}


int2 __device__ d_get_coords(int x, int y, int w, int h, int width, int height) {
    const auto pixel_x = int(x / float(w) * width);
    const auto pixel_y = int(y / float(w) * height);
    //int cord[] = { pixel_x, pixel_y };
    return make_int2(pixel_x, pixel_y);
}

void __device__ brushBasicPixel(int x, int y, int mx, int my, int w, int h,
                                float *buffer_height, float3 *buffer_color, const BrushSettings &bs) {

    float radius = sqrtf((x - mx) * (x - mx) + (y - my) * (y - my));
    float brush_radius = bs.size / 2.0f;

    int i = d_getBufferIndex(x, y, w);

    // paint color
    float strength = bs.pressure * d_cosine_fallof(radius / brush_radius, bs.falloff);
    float3 color = d_interpolate_color(buffer_color[i], strength, bs.color);
    buffer_color[i] = color;

    // paint height
    strength = bs.heightPressure * d_cosine_fallof(radius / brush_radius, bs.falloff);
    buffer_height[i] = clamp(buffer_height[i] + strength, -1.0f, 1.0f);
}

__device__
void updateDisplayPixel(int x, int y, int w, int h, uchar4 *buffer_pbo, float *buffer_height, float3 *buffer_color) {
    int i = d_getBufferIndex(x, y, w);

    auto normal = d_getNormal(x, y, w, h, buffer_height);

    float3 lighting = normalize(make_float3(0.07f, 0.07f, 1.0f));

    // TODO: use lighting vector here
    float shadow =
            normal.z * 0.80f - normal.x * 0.1f - normal.y * 0.1f + (d_sampleHeight(x, y, w, h, buffer_height)) / 4.0f;
    shadow = clamp(shadow, 0.0f, 1.0f);

    float specular = 1.0f - length(normal - lighting);
    specular = powf(specular, 8.0f);
    specular = clamp(specular, 0.0f, 1.0f);

    float3 color = lerp(buffer_color[i] * shadow, make_float3(255.0f), specular);

    // view normals (TODO: remove or make normals visualization feature)
    /*color.x = normal.x * 255.0 / 2 + 255.0 / 2;
    color.y = normal.y * 255.0 / 2 + 255.0 / 2;
    color.z = normal.z * 255;*/
    //color = clamp(color, make_float3(0.0f), make_float3(255.0f));
    buffer_pbo[i] = make_uchar4(color.x, color.y, color.z, 0);
}

// kernels
// Kernel that paints brush basic
__global__
void brushBasicKernel(uchar4 *pbo, float *buffer_height, float3 *buffer_color,
                      int width, int height, int mx, int my, const BrushSettings bs) {

    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    float radius = sqrtf((x - mx) * (x - mx) + (y - my) * (y - my));
    float brush_radius = bs.size / 2.0f;
    if (radius < brush_radius) {
        if (in_bounds(x, y, width, height)) {
            // use brush
            brushBasicPixel(x, y, mx, my, width, height, buffer_height, buffer_color, bs);
        }
    }
}

// Kernel that updates image display
__global__
void updateDisplayKernel(uchar4 *pbo, float *buffer_height, float3 *buffer_color,
                         int width, int height, int mx, int my, const BrushSettings bs) {

    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    float radius = sqrtf((x - mx) * (x - mx) + (y - my) * (y - my));
    float brush_radius = bs.size / 2.0f;

    if (radius < brush_radius && in_bounds(x, y, width, height)) {
        // shading pixels
        updateDisplayPixel(x, y, width, height, pbo, buffer_height, buffer_color);
    }
}

void GPUPainter::setDimensions(int w1, int h1, uchar4 *pbo) {
    w = w1;
    h = h1;

    buffer_pbo = pbo;

    int buf_size = w * h;

    printf("init/resize gpu buffers (%d, %d)\n", w, h);

    checkCudaErrors(cudaFree(buffer_color));
    checkCudaErrors(cudaFree(buffer_height));

    checkCudaErrors(cudaMalloc((void **) &buffer_height, buf_size * sizeof(float)));
    checkCudaErrors(cudaMemset(buffer_height, 0, buf_size * sizeof(float)));

    checkCudaErrors(cudaMalloc((void **) &buffer_color, buf_size * sizeof(float3)));
    // @ TODO init buffer with correct color
    checkCudaErrors(cudaMemset(buffer_color, 0, buf_size * sizeof(float3)));

}

void GPUPainter::setBrushType(BrushType type) {
    using namespace std::placeholders;
    switch (type) {
        case BrushType::Default:
            paint_function = std::bind(&GPUPainter::brushBasic, this, _1, _2);
            break;
        case BrushType::Textured:
            paint_function = std::bind(&GPUPainter::brushTextured, this, _1, _2);
            break;
        case BrushType::Third:
            std::clog << "Warning: chose unused brush\n";
            break;
        default:
            throw std::runtime_error("Invalid brush type: "
                                     + std::to_string(static_cast<int>(type)));
    }
}

void
GPUPainter::setTexture(const std::string &type, const unsigned char *data, int width, int height, int bytes_per_pixel) {
    int pixel_datasize = width * height * bytes_per_pixel * sizeof(unsigned char);

    if (type == "colorFilename") {
        checkCudaErrors(cudaMalloc((void **) &d_color_texture, pixel_datasize));
        checkCudaErrors(cudaMemcpy(d_color_texture, data, pixel_datasize, cudaMemcpyHostToDevice));
        color_image_height = height;
        color_image_width = width;
        color_image_bytes_per_pixel = bytes_per_pixel;
    } else {
        checkCudaErrors(cudaMalloc((void **) &d_height_texture, pixel_datasize));
        checkCudaErrors(cudaMemcpy(d_height_texture, data, pixel_datasize, cudaMemcpyHostToDevice));
        height_image_height = height;
        height_image_width = width;
        height_image_bytes_per_pixel = bytes_per_pixel;
    }
}

void GPUPainter::doPainting(int x, int y, uchar4 *pbo) {
    paint_function(x, y);
    //std::clog << "[GPU] Painting: " << painting_duration / 1e6f << "ms\n";
}

// brush functions
void GPUPainter::brushBasic(int mx, int my) {
    const int blockSideLength = 32;
    const dim3 blockSize(blockSideLength, blockSideLength);
    const dim3 blocksPerGrid(
            (w + blockSize.x - 1) / blockSize.x,
            (h + blockSize.y - 1) / blockSize.y);
    // @ TODO compute cuda time
    brushBasicKernel << < blocksPerGrid, blockSize >> >
                                         (buffer_pbo, buffer_height, buffer_color, w, h, mx, my, brushSettings);
    updateDisplayKernel << < blocksPerGrid, blockSize >> >
                                            (buffer_pbo, buffer_height, buffer_color, w, h, mx, my, brushSettings);
    checkCudaErrors(cudaDeviceSynchronize());
}


void __device__ brushTexturedPixel(int x, int y, int mx, int my, int w, int h,
                                   float *buffer_height, float3 *buffer_color, const BrushSettings &bs,
                                   unsigned char *color_texture, unsigned char *height_texture, int cih, int ciw,
                                   int hih, int hiw, int cbpp, int hbpp) {

    float maxRadius = bs.size / 2;

    float radius = sqrtf((x - mx) * (x - mx) + (y - my) * (y - my));

    int i = d_getBufferIndex(x, y, w);

    float strength = bs.pressure * d_cosine_fallof(radius / maxRadius, bs.falloff);
    const auto color_coords =
            d_get_coords(x - mx + maxRadius, y - my + maxRadius, maxRadius * 2, maxRadius * 2, ciw, cih);
    const auto pixel = d_getImageByteIndex(color_coords.x, color_coords.y, cih, ciw, cbpp);
    buffer_color[i] = d_interpolate_color(
            buffer_color[i],
            strength,
            make_float3(color_texture[pixel], color_texture[pixel + 1], color_texture[pixel + 2]));

    const auto height_coords =
            d_get_coords(x - mx + maxRadius, y - my + maxRadius, maxRadius * 2, maxRadius * 2, hiw, hih);
    const auto height = color_texture[d_getImageByteIndex(height_coords.x, height_coords.y, hih, hiw, hbpp)] * 0.001f;
    strength = bs.heightPressure * height * d_cosine_fallof(radius / maxRadius, bs.falloff);
    buffer_height[i] = clamp(buffer_height[i] + strength, -1.0f, 1.0f);
}

__global__
void brushTexturedKernel(uchar4 *pbo, float *buffer_height, float3 *buffer_color,
                         int width, int height, int mx, int my, const BrushSettings bs, unsigned char *color_texture,
                         unsigned char *height_texture, int cih, int ciw, int hih, int hiw, int cbpp, int hbpp) {

    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    float radius = sqrtf((x - mx) * (x - mx) + (y - my) * (y - my));
    float brush_radius = bs.size / 2.0f;

    if (radius < brush_radius && in_bounds(x, y, width, height)) {
        // shading pixels
        brushTexturedPixel(x, y, mx, my, width, height, buffer_height, buffer_color, bs, color_texture, height_texture,
                           cih, ciw, hih, hiw, cbpp, hbpp);
    }
}


void GPUPainter::brushTextured(int mx, int my) {
    const int blockSideLength = 32;
    const dim3 blockSize(blockSideLength, blockSideLength);
    const dim3 blocksPerGrid(
            (w + blockSize.x - 1) / blockSize.x,
            (h + blockSize.y - 1) / blockSize.y);

    if (d_color_texture == nullptr || d_height_texture == nullptr) {
        return;
    }

    brushTexturedKernel << < blocksPerGrid, blockSize >> >
                                            (buffer_pbo, buffer_height, buffer_color, w, h, mx, my, brushSettings, d_color_texture, d_height_texture, color_image_height, color_image_width, height_image_height, height_image_width, color_image_bytes_per_pixel, height_image_bytes_per_pixel);
    updateDisplayKernel << < blocksPerGrid, blockSize >> >
                                            (buffer_pbo, buffer_height, buffer_color, w, h, mx, my, brushSettings);
    checkCudaErrors(cudaDeviceSynchronize());
}



