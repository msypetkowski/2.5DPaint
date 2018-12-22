#pragma once

#include "cuda_runtime.h"

#include "color.h"

/*
 * Helper structure for passing parameters to gpu painter kernels
 */
struct KernelArgs {

    int w, h; // currently painted image width and height

    uchar4* pbo = nullptr; // device pixel buffer object global memory pointer

    float3* buff_color_dptr = nullptr; // device color buffer global memory pointer
    float* buff_height_dptr = nullptr; // device height buffer global memory pointer


    int ctex_width, ctex_height, htex_height, htex_width; // color and height textures sizes

    int ctex_bpp, htex_bpp; // color and height texture bytes per pixel

    unsigned char* ctex_dptr = nullptr; // color texture data global memory device pointer
    unsigned char* htex_dptr = nullptr; // height texture data global memory device pointer


    dim3 blockSize, blocksPerGrid; // cuda kernels launch parameters
};
