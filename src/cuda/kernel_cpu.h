#pragma once
#include "cuda_runtime.h"

#include <QVector>

#include "../brush.h"

// these buffers are used when CUDA disabled

// buffer for display information
extern QVector<uchar4> cpu_buffer;

// internal representation buffers
extern QVector<uchar4> cpu_buffer_color;
extern QVector<qreal> cpu_buffer_height;

void update_whole_display(int w1, int h1);
void brush_basic(int w, int h, int mx, int my, const BrushSettings&);
