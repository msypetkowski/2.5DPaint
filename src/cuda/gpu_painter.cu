#include "gpu_painter.h"

#include <iostream>
#include <chrono>
#include <vector>

#include "cuda_runtime.h"
#include "helper_cuda.h"
#include "helper_math.h"
#include "device_launch_parameters.h"

#include "utils.h"

// reinitialize height, color and display buffers
void GPUPainter::setDimensions(int w1, int h1, uchar4 *pbo) {

    // update image width and height properties
    args.w = w = w1;
    args.h = h = h1;

    args.pbo = buffer_pbo = pbo; // update device pixel buffer object memory pointer

    int buf_size = w * h; // calculate number of pixels

    printf("[GPU] init/resize GPU buffers (%d, %d)\n", w, h);

    // free old buffers if there was any allocated
    if (buffer_color) { checkCudaErrors(cudaFree(buffer_color)); buffer_color = nullptr; }
    if (buffer_height) { checkCudaErrors(cudaFree(buffer_height)); buffer_height = nullptr; }

    // allocate device buffers memory
    checkCudaErrors(cudaMalloc((void **) &buffer_height, buf_size * sizeof(float)));
    //checkCudaErrors(cudaMemset(buffer_height, 0, buf_size * sizeof(float)));

    checkCudaErrors(cudaMalloc((void **) &swap_buffer_height, buf_size * sizeof(float)));
    //checkCudaErrors(cudaMemset(buffer_height, 0, buf_size * sizeof(float)));

    checkCudaErrors(cudaMalloc((void **) &buffer_color, buf_size * sizeof(float3)));
    //checkCudaErrors(cudaMemset(buffer_color, 0, buf_size * sizeof(float3)));

    // update kernel args, this structure will be later passed to cuda kernels
    args.buff_color_dptr = buffer_color;
    args.buff_height_dptr = buffer_height;
    args.swap_buff_height_dptr = swap_buffer_height;

    args.light_direction = lightDirection;

    // set cuda kernels launch params
    const int blockSideLength = 32;

    args.blockSize = dim3(blockSideLength, blockSideLength);
    args.blocksPerGrid = dim3((w + args.blockSize.x - 1) / args.blockSize.x, (h + args.blockSize.y - 1) / args.blockSize.y);

    // clear image to default color and height
    Painter::clear();
}

// assign current paint function depending on brush type
void GPUPainter::setBrushType(BrushType type) {
    using namespace std::placeholders;
    switch (type) {
        case BrushType::Default:
            paint_function = std::bind(&GPUPainter::brushBasic, this, _1, _2);
            break;
        case BrushType::Textured:
            paint_function = std::bind(&GPUPainter::brushTextured, this, _1, _2);
            break;
        case BrushType::Smooth:
            paint_function = std::bind(&GPUPainter::brushSmooth, this, _1, _2);
            break;
        default:
            throw std::runtime_error("Invalid brush type: "
                                     + std::to_string(static_cast<int>(type)));
    }
}

// assign current height or color texture
void GPUPainter::setTexture(const std::string &type, const unsigned char *data, int width, int height, int bytes_per_pixel) {
    int pixel_datasize = width * height * bytes_per_pixel * sizeof(unsigned char);

    unsigned char *d_color_texture = nullptr, *d_height_texture = nullptr;

    if (type == "colorFilename") { // assign color texture
        checkCudaErrors(cudaMalloc((void **) &d_color_texture, pixel_datasize));
        checkCudaErrors(cudaMemcpy(d_color_texture, data, pixel_datasize, cudaMemcpyHostToDevice));
        args.ctex_height = height;
        args.ctex_width = width;
        args.ctex_bpp = bytes_per_pixel;
        args.ctex_dptr = d_color_texture;

    } else { // assign height texture
        checkCudaErrors(cudaMalloc((void **) &d_height_texture, pixel_datasize));
        checkCudaErrors(cudaMemcpy(d_height_texture, data, pixel_datasize, cudaMemcpyHostToDevice));
        args.htex_height = height;
        args.htex_width = width;
        args.htex_bpp = bytes_per_pixel;
        args.htex_dptr = d_height_texture;
    }
}

// wrapper function for executing currently assigned paint function
void GPUPainter::doPainting(int x, int y, uchar4 *pbo) {
    auto start_time = std::chrono::steady_clock::now();
    paint_function(x, y);
    auto end_time = std::chrono::steady_clock::now();

    std::clog << "[GPU] Painting time: " <<
         (float)std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count() / (float)1e6 << " ms\n";
}

// sets image to default state
void GPUPainter::clearImage(float3 color, float height) {

    // prepare temporary buffers on cpu
    std::vector<float> fillh(w * h);
    std::fill(fillh.begin(), fillh.end(), height);

    std::vector<float3> fillcol(w * h);
    std::fill(fillcol.begin(), fillcol.end(), color);

    // copy cpu buffer content to gpu memory
    if (args.buff_height_dptr != nullptr) {
        checkCudaErrors(cudaMemcpy(args.buff_height_dptr, fillh.data(), w * h * sizeof(float), cudaMemcpyHostToDevice));
    }
    if (args.buff_height_dptr != nullptr) {
        checkCudaErrors(cudaMemcpy(args.swap_buff_height_dptr, fillh.data(), w * h * sizeof(float), cudaMemcpyHostToDevice));
    }
    if (args.buff_color_dptr != nullptr) {
        checkCudaErrors(cudaMemcpy(args.buff_color_dptr, fillcol.data(), w * h * sizeof(float3), cudaMemcpyHostToDevice));
    }

    std::clog << "[GPU] Clear image\n";
}

/**********************************************************************************************************************/
/*
 * Kernels helper functions
 */
bool __device__ inBounds(int x, int y, int w, int h) {
    return x >= 0 && x < w && y >= 0 && y < h;
}

// helper function for getting flat array index depend on pixel x, y, and image width (w) parameters
int __device__ getBufferIndex(int x, int y, int w) { return (w - 1 - x + (y * w)); };

// helper function for getting flat texture data array index
int __device__ getTextureByteIndex(int x, int y, int w, int h, int bpp) { return (w * bpp) * y + x * bpp; };

// helper function for getting height value for given pixel coordinates
float __device__ sampleHeight(int x, int y, const KernelArgs &args) {
    x = clamp(x, 0, args.w - 1);
    y = clamp(y, 0, args.h - 1);
    return args.buff_height_dptr[getBufferIndex(x, y, args.w)];
}

// helper function for calculating normal vector for given pixel coords
float3 __device__ getNormal(int x, int y, float bending, const KernelArgs &args) {
    float dx = 0.0f, dy = 0.0f;

    // get height values near pixel given by x, y coords
    //
    //        |   top  |
    //  ______|________|_______
    //   left |   mid  | right
    //  ______|________|_______
    //        | bottom |
    //        |        |

    auto mid = sampleHeight(x, y, args);
    auto left = sampleHeight(x - 1, y, args);
    auto right = sampleHeight(x + 1, y, args);
    auto top = sampleHeight(x, y + 1, args);
    auto bottom = sampleHeight(x, y - 1, args);

    dx += normal_from_delta(mid - right) / 2;
    dx -= normal_from_delta(mid - left) / 2;

    dy += normal_from_delta(mid - top) / 2;
    dy -= normal_from_delta(mid - bottom) / 2;

    // scale normals along x, y axis to adjust them for proper user display
    dx *= bending;
    dy *= bending;

    dx = dx / sqrtf(dx * dx + dy * dy + 1);
    dy = dy / sqrtf(dx * dx + dy * dy + 1);

    auto ret = make_float3(dx, dy, sqrtf(fabsf(1.0f - dx * dx - dy * dy)));
    return normalize(ret);
}

// swap height buffer pointers (double buffering feature)
void GPUPainter::swapHeightBuffer() {
    float* tmp = buffer_height;
    buffer_height = swap_buffer_height;
    swap_buffer_height = tmp;

    args.buff_height_dptr = buffer_height;
    args.swap_buff_height_dptr = swap_buffer_height;
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

    // current brush radius
    float brush_radius = bs.size / 2.0f;

    // calculate pixel coords, kernel is launched only for brush pixels not for whole image
    // thats why mouse coordinates (mx, my) have to be added here
    int x = (blockIdx.x * blockDim.x) + threadIdx.x + mx - int(brush_radius);
    int y = (blockIdx.y * blockDim.y) + threadIdx.y + my - int(brush_radius);

    // calculate current pixel distance from brush middle point
    float radius = sqrtf((x - mx) * (x - mx) + (y - my) * (y - my));

    if (radius < brush_radius) {
        if (inBounds(x, y, args.w, args.h)) { // check if current pixel is within image bounds
            int i = getBufferIndex(x, y, args.w); // get flat buffer index for current pixel

            // calculate brush color strength
            float strength = bs.pressure * cosine_fallof(radius / brush_radius, bs.falloff);

            // interpolate color depend on brush strength
            float3 color = interpolate_color(args.buff_color_dptr[i], strength, bs.color);

            // store new color in color buffer
            args.buff_color_dptr[i] = color;

            // calculate brush height strength
            strength = bs.heightPressure * cosine_fallof(radius / brush_radius, bs.falloff);

            // calculate current pixel height
            float result = clamp(args.buff_height_dptr[i] + strength, -1.0f, 1.0f);

            // store height in 2 buffers (swap buffer also has to be updated because it is used in smooth brush later)
            args.buff_height_dptr[i] = result;
            args.swap_buff_height_dptr[i] = result;
        }
    }
}

/*
 * Textured brush kernel
 */
__global__
void brushTextured_GPU_KERNEL(int mx, int my, const BrushSettings bs, const KernelArgs args) {

    float brush_radius = bs.size / 2.0f;

    // calculate pixel coords, kernel is launched only for brush pixels not for whole image
    // thats why mouse coordinates (mx, my) have to be added here
    int x = (blockIdx.x * blockDim.x) + threadIdx.x + mx - int(brush_radius);
    int y = (blockIdx.y * blockDim.y) + threadIdx.y + my - int(brush_radius);

    // calculate current pixel distance from brush middle point
    float radius = sqrtf((x - mx) * (x - mx) + (y - my) * (y - my));

    if (radius < brush_radius && inBounds(x, y, args.w, args.h)) {
        int i = getBufferIndex(x, y, args.w);

        // calculate brush strength for color
        float strength = bs.pressure * cosine_fallof(radius / brush_radius, bs.falloff);

        // get color texture pixel coords depend on current image pixel coordinates
        const auto color_coords = get_coords(x - mx + brush_radius,
                                             y - my + brush_radius,
                                             bs.size,
                                             bs.size,
                                             args.ctex_width,
                                             args.ctex_height);

        // get texture data buffer pixel index
        const auto pixel = getTextureByteIndex( color_coords.x,
                                                color_coords.y,
                                                args.ctex_width,
                                                args.ctex_height,
                                                args.ctex_bpp);

        args.buff_color_dptr[i] = interpolate_color(args.buff_color_dptr[i],
                                                    strength,
                                                    make_float3(args.ctex_dptr[pixel],
                                                                args.ctex_dptr[pixel + 1],
                                                                args.ctex_dptr[pixel + 2]));

        // get height texture pixel coords depend on current image pixel coordinates
        const auto height_coords = get_coords(  x - mx + brush_radius,
                                                y - my + brush_radius,
                                                bs.size,
                                                bs.size,
                                                args.htex_width,
                                                args.htex_height);

        // get height texture value for computed pixel
        const auto height = args.htex_dptr[getTextureByteIndex(height_coords.x,
                                           height_coords.y,
                                           args.htex_width,
                                           args.htex_height,
                                           args.htex_bpp)] * 0.001f;

        // calculate brush strength for height
        strength = bs.heightPressure * height * cosine_fallof(radius / brush_radius, bs.falloff);

        float result = clamp(args.buff_height_dptr[i] + strength, -1.0f, 1.0f);

        // store height in 2 buffers (swap buffer also has to be updated because it is used in smooth brush later)
        args.buff_height_dptr[i] = result;
        args.swap_buff_height_dptr[i] = result;
    }
}


/*
 * Smooth brush kernel
 */
__global__
void brushSmooth_GPU_KERNEL(int mx, int my, const BrushSettings bs, const KernelArgs args) {

    float brush_radius = bs.size / 2.0f;

    // calculate pixel coords, kernel is launched only for brush pixels not for whole image
    // thats why mouse coordinates (mx, my) have to be added here
    int x = (blockIdx.x * blockDim.x) + threadIdx.x + mx - int(brush_radius);
    int y = (blockIdx.y * blockDim.y) + threadIdx.y + my - int(brush_radius);

    float radius = sqrtf((x - mx) * (x - mx) + (y - my) * (y - my));

    if (radius < brush_radius) {
        if (inBounds(x, y, args.w, args.h)) {
            int i = getBufferIndex(x, y, args.w);

            // calculate strength of smoothing
            float strength = cosine_fallof(radius / brush_radius, bs.falloff);

            // apply mean convolution filter 3x3
            auto mid = sampleHeight(x, y, args);

            auto left = sampleHeight(x - 1, y, args);
            auto right = sampleHeight(x + 1, y, args);

            auto top = sampleHeight(x, y + 1, args);
            auto bottom = sampleHeight(x, y - 1, args);

            auto topleft = sampleHeight(x - 1, y - 1, args);
            auto topright = sampleHeight(x + 1, y - 1, args);
            auto bottomleft = sampleHeight(x - 1, y + 1, args);
            auto bottomright = sampleHeight(x + 1, y + 1, args);

            float result = (mid + left + right + top + bottom + topleft + topright + bottomleft + bottomright) / 9.0f;

            // depending on brush smoothing strength
            args.swap_buff_height_dptr[i] = mid + strength * (result - mid);
        }
    }
}


/*
 * Kernel that updates image display
 * This kernel calculates normals based on height buffer and shades pixels properly
 * Result color is stored in pbo buffer which is rendered on the screen using OpenGL (QOpenGLWidget)
 */
__global__
void updateDisplay_GPU_KERNEL(int mx, int my, const BrushSettings bs, const KernelArgs args) {

    float brush_radius = bs.size / 2.0f;

    bool update_whole_display = mx == -1 && my == -1;

    int x = (blockIdx.x * blockDim.x) + threadIdx.x + (update_whole_display ? 0 : (mx - int(brush_radius)));
    int y = (blockIdx.y * blockDim.y) + threadIdx.y + (update_whole_display ? 0 : (my - int(brush_radius)));

    if (inBounds(x, y, args.w, args.h)) {
        // shading pixels
        int i = getBufferIndex(x, y, args.w);

        auto normal = getNormal(x, y, bs.normalBending, args);

        float3 color;

        if (!bs.renderNormals) {

            float3 lighting = normalize(args.light_direction);

            float shadow = fabsf(dot(lighting, normal));
            shadow = clamp(shadow, 0.0f, 1.0f);

            float specular = 1.0f - length(normal - lighting);
            specular = powf(specular, 8.0f);
            specular = clamp(specular, 0.0f, 1.0f);

            color = lerp(args.buff_color_dptr[i] * shadow, make_float3(255.0f), specular);
        } else {
            // view normals
            color.x = normal.x * 255.0 / 2 + 255.0 / 2;
            color.y = normal.y * 255.0 / 2 + 255.0 / 2;
            color.z = normal.z * 255;
        }
        color = clamp(color, make_float3(0.0f), make_float3(255.0f));
        args.pbo[i] = make_uchar4(color.x, color.y, color.z, 0);
    }
}

/**********************************************************************************************************************/

/*
 * GPU painter kernels launch functions
 */


void GPUPainter::brushBasic(int mx, int my) {

    int size = int(brushSettings.size);

    // set cuda kernels launch params
    const int blockSideLength = 32;

    args.blockSize = dim3(blockSideLength, blockSideLength);
    args.blocksPerGrid = dim3((size + args.blockSize.x - 1) / args.blockSize.x, (size + args.blockSize.y - 1) / args.blockSize.y);

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

    int size = int(brushSettings.size);

    // set cuda kernels launch params
    const int blockSideLength = 32;

    args.blockSize = dim3(blockSideLength, blockSideLength);
    args.blocksPerGrid = dim3((size + args.blockSize.x - 1) / args.blockSize.x, (size + args.blockSize.y - 1) / args.blockSize.y);

    brushTextured_GPU_KERNEL << < args.blocksPerGrid, args.blockSize >> >(mx, my, brushSettings, args);
    updateDisplay_GPU_KERNEL << < args.blocksPerGrid, args.blockSize >> >(mx, my, brushSettings, args);
    checkCudaErrors(cudaDeviceSynchronize());
}


void GPUPainter::brushSmooth(int mx, int my) {
    int size = int(brushSettings.size);

    // set cuda kernels launch params
    const int blockSideLength = 32;

    args.blockSize = dim3(blockSideLength, blockSideLength);
    args.blocksPerGrid = dim3((size + args.blockSize.x - 1) / args.blockSize.x, (size + args.blockSize.y - 1) / args.blockSize.y);

    brushSmooth_GPU_KERNEL << < args.blocksPerGrid, args.blockSize >> >(mx, my, brushSettings, args);
    updateDisplay_GPU_KERNEL << < args.blocksPerGrid, args.blockSize >> >(mx, my, brushSettings, args);
    checkCudaErrors(cudaDeviceSynchronize());

    swapHeightBuffer();
}


void GPUPainter::updateWholeDisplay() {

    // set cuda kernels launch params
    const int blockSideLength = 32;

    args.blockSize = dim3(blockSideLength, blockSideLength);
    args.blocksPerGrid = dim3((w + args.blockSize.x - 1) / args.blockSize.x, (h + args.blockSize.y - 1) / args.blockSize.y);

    updateDisplay_GPU_KERNEL << < args.blocksPerGrid, args.blockSize >> >(-1, -1, brushSettings, args);
    checkCudaErrors(cudaDeviceSynchronize());
}




