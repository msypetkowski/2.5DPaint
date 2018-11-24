#include "cuda_runtime.h"

extern uchar4 *pbo_dptr;

void setupCuda();
int cuda_main(int w, int h, int mx, int my);
