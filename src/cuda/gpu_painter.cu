#include "gpu_painter.h"

#include <iostream>
#include <chrono>

#include "cuda_runtime.h"
#include "helper_cuda.h"
#include "helper_math.h"
#include "device_launch_parameters.h"

#include "utils.h"


void GPUPainter::setDimensions(int w1, int h1, uchar4 *pbo) {
    args.w = w = w1;
    args.h = h = h1;

    args.pbo = buffer_pbo = pbo;

    int buf_size = w * h;

    printf("[GPU] init/resize GPU buffers (%d, %d)\n", w, h);

    if (buffer_color) { checkCudaErrors(cudaFree(buffer_color)); buffer_color = nullptr; }
    if (buffer_height) { checkCudaErrors(cudaFree(buffer_height)); buffer_height = nullptr; }

    checkCudaErrors(cudaMalloc((void **) &buffer_height, buf_size * sizeof(float)));
    checkCudaErrors(cudaMemset(buffer_height, 0, buf_size * sizeof(float)));

    checkCudaErrors(cudaMalloc((void **) &buffer_color, buf_size * sizeof(float3)));
    // @ TODO init buffer with correct color
    checkCudaErrors(cudaMemset(buffer_color, 0, buf_size * sizeof(float3)));

    args.buff_color_dptr = buffer_color;
    args.buff_height_dptr = buffer_height;

    // set cuda kernels launch params
    const int blockSideLength = 32;

    args.blockSize = dim3(blockSideLength, blockSideLength);
    args.blocksPerGrid = dim3((w + args.blockSize.x - 1) / args.blockSize.x, (h + args.blockSize.y - 1) / args.blockSize.y);
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
            std::clog << "[GPU] Warning: chose unused brush\n";
            break;
        default:
            throw std::runtime_error("Invalid brush type: "
                                     + std::to_string(static_cast<int>(type)));
    }
}

void GPUPainter::setTexture(const std::string &type, const unsigned char *data, int width, int height, int bytes_per_pixel) {
    int pixel_datasize = width * height * bytes_per_pixel * sizeof(unsigned char);

    unsigned char *d_color_texture = nullptr, *d_height_texture = nullptr;

    if (type == "colorFilename") {
        checkCudaErrors(cudaMalloc((void **) &d_color_texture, pixel_datasize));
        checkCudaErrors(cudaMemcpy(d_color_texture, data, pixel_datasize, cudaMemcpyHostToDevice));
        args.ctex_height = height;
        args.ctex_width = width;
        args.ctex_bpp = bytes_per_pixel;
        args.ctex_dptr = d_color_texture;
    } else {
        checkCudaErrors(cudaMalloc((void **) &d_height_texture, pixel_datasize));
        checkCudaErrors(cudaMemcpy(d_height_texture, data, pixel_datasize, cudaMemcpyHostToDevice));
        args.htex_height = height;
        args.htex_width = width;
        args.htex_bpp = bytes_per_pixel;
        args.htex_dptr = d_height_texture;
    }
}

void GPUPainter::doPainting(int x, int y, uchar4 *pbo) {
    auto start_time = std::chrono::steady_clock::now();
    paint_function(x, y);
    auto end_time = std::chrono::steady_clock::now();

    std::clog << "[GPU] Painting time: " <<
         (float)std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count() / (float)1e6 << " ms\n";
}

/**********************************************************************************************************************/
/*
 * Kernels helper functions
 */
bool __device__ inBounds(int x, int y, int w, int h) {
    return x >= 0 && x < w && y >= 0 && y < h;
}

int __device__ getBufferIndex(int x, int y, int w) { return (w - 1 - x + (y * w)); };

int __device__ getTextureByteIndex(int x, int y, int h, int w, int bpp) { return (w * bpp) * y + x * bpp; };

float __device__ sampleHeight(int x, int y, const KernelArgs &args) {
    x = clamp(x, 0, args.w - 1);
    y = clamp(y, 0, args.h - 1);
    return args.buff_height_dptr[getBufferIndex(x, y, args.w)];
}

float3 __device__ getNormal(int x, int y, const KernelArgs &args) {
    float dx = 0.0f, dy = 0.0f;

    auto mid = sampleHeight(x, y, args);
    auto left = sampleHeight(x - 1, y, args);
    auto right = sampleHeight(x + 1, y, args);
    auto top = sampleHeight(x, y + 1, args);
    auto bottom = sampleHeight(x, y - 1, args);

    dx += normal_from_delta(mid - right) / 2;
    dx -= normal_from_delta(mid - left) / 2;

    dy += normal_from_delta(mid - top) / 2;
    dy -= normal_from_delta(mid - bottom) / 2;

    // TODO: make parameter or constant
    dx *= 100;
    dy *= 100;

    dx = dx / sqrtf(dx * dx + dy * dy + 1);
    dy = dy / sqrtf(dx * dx + dy * dy + 1);

    auto ret = make_float3(dx, dy, sqrtf(clamp(1.0f - dx * dx - dy * dy, 0.0f, 1.0f)));
    return normalize(ret);
}
// @TODO refactor
int2 __device__ d_get_coords(int x, int y, int w, int h, int width, int height) {
    const auto pixel_x = int(x / float(w) * width);
    const auto pixel_y = int(y / float(w) * height);
    //int cord[] = { pixel_x, pixel_y };
    return make_int2(pixel_x, pixel_y);
}


/**********************************************************************************************************************/
/*
 * CUDA kernels used for brush painting and display updating
 */

/*
 * Basic brush kernel
 */
__global__
void brushBasic_GPU_KERNEL(int mx, int my, const BrushSettings bs, const KernelArgs args) {

    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    float radius = sqrtf((x - mx) * (x - mx) + (y - my) * (y - my));
    float brush_radius = bs.size / 2.0f;
    if (radius < brush_radius) {
        if (inBounds(x, y, args.w, args.h)) {
            // use brush

            int i = getBufferIndex(x, y, args.w);

            // paint color
            float strength = bs.pressure * cosine_fallof(radius / brush_radius, bs.falloff);
            float3 color = interpolate_color(args.buff_color_dptr[i], strength, bs.color);
            args.buff_color_dptr[i] = color;

            // paint height
            strength = bs.heightPressure * cosine_fallof(radius / brush_radius, bs.falloff);
            args.buff_height_dptr[i] = clamp(args.buff_height_dptr[i] + strength, -1.0f, 1.0f);
        }
    }
}

/*
 * Textured brush kernel
 */
__global__
void brushTextured_GPU_KERNEL(int mx, int my, const BrushSettings bs, const KernelArgs args) {

    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    float radius = sqrtf((x - mx) * (x - mx) + (y - my) * (y - my));
    float brush_radius = bs.size / 2.0f;

    if (radius < brush_radius && inBounds(x, y, args.w, args.h)) {
        float maxRadius = bs.size / 2.0f;

        int i = getBufferIndex(x, y, args.w);

        float strength = bs.pressure * cosine_fallof(radius / maxRadius, bs.falloff);
        const auto color_coords = d_get_coords(x - mx + maxRadius, y - my + maxRadius, maxRadius * 2, maxRadius * 2, args.ctex_width, args.ctex_height);
        const auto pixel = getTextureByteIndex(color_coords.x, color_coords.y, args.ctex_width, args.ctex_height,
                                               args.ctex_bpp);
        args.buff_color_dptr[i] = interpolate_color(
                args.buff_color_dptr[i],
                strength,
                make_float3(args.ctex_dptr[pixel], args.ctex_dptr[pixel + 1], args.ctex_dptr[pixel + 2]));

        const auto height_coords = d_get_coords(x - mx + maxRadius, y - my + maxRadius, maxRadius * 2, maxRadius * 2, args.htex_width, args.htex_height);
        const auto height = args.ctex_dptr[getTextureByteIndex(height_coords.x, height_coords.y, args.htex_width, args.htex_height, args.htex_bpp)] * 0.001f;
        strength = bs.heightPressure * height * cosine_fallof(radius / maxRadius, bs.falloff);
        args.buff_height_dptr[i] = clamp(args.buff_height_dptr[i] + strength, -1.0f, 1.0f);
    }
}


/*
 * Kernel that updates image display
 * This kernel calculates normals based on height buffer and shades pixels properly
 * Result color is stored in pbo buffer which is rendered on the screen using OpenGL (QOpenGLWidget)
 */
__global__
void updateDisplay_GPU_KERNEL(int mx, int my, const BrushSettings bs, const KernelArgs args) {

    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    float radius = sqrtf((x - mx) * (x - mx) + (y - my) * (y - my));
    float brush_radius = bs.size / 2.0f;

    if (radius < brush_radius && inBounds(x, y, args.w, args.h)) {
        // shading pixels
        int i = getBufferIndex(x, y, args.w);

        auto normal = getNormal(x, y, args);

        float3 lighting = normalize(make_float3(0.07f, 0.07f, 1.0f));

        // TODO: use lighting vector here
        float shadow = normal.z * 0.80f - normal.x * 0.1f - normal.y * 0.1f + (sampleHeight(x, y, args)) / 4.0f;
        shadow = clamp(shadow, 0.0f, 1.0f);

        float specular = 1.0f - length(normal - lighting);
        specular = powf(specular, 8.0f);
        specular = clamp(specular, 0.0f, 1.0f);

        float3 color = lerp(args.buff_color_dptr[i] * shadow, make_float3(255.0f), specular);

        // view normals (TODO: remove or make normals visualization feature)
        /*color.x = normal.x * 255.0 / 2 + 255.0 / 2;
        color.y = normal.y * 255.0 / 2 + 255.0 / 2;
        color.z = normal.z * 255;*/
        //color = clamp(color, make_float3(0.0f), make_float3(255.0f));
        args.pbo[i] = make_uchar4(color.x, color.y, color.z, 0);
    }
}

/**********************************************************************************************************************/

/*
 * GPU painter kernels launch functions
 */


void GPUPainter::brushBasic(int mx, int my) {

    // @ TODO compute real cuda time
    brushBasic_GPU_KERNEL <<< args.blocksPerGrid, args.blockSize >>>(mx, my, brushSettings, args);
    updateDisplay_GPU_KERNEL <<< args.blocksPerGrid, args.blockSize >>>(mx, my, brushSettings, args);
    checkCudaErrors(cudaDeviceSynchronize());
}


void GPUPainter::brushTextured(int mx, int my) {
    if (args.ctex_dptr == nullptr || args.htex_dptr == nullptr) {
        std::clog << "[GPU] warning: textures are not set\n";
        return;
    }

    brushTextured_GPU_KERNEL << < args.blocksPerGrid, args.blockSize >> >(mx, my, brushSettings, args);
    updateDisplay_GPU_KERNEL << < args.blocksPerGrid, args.blockSize >> >(mx, my, brushSettings, args);
    checkCudaErrors(cudaDeviceSynchronize());
}




