# 2.5DPaint
2.5D painting software using NVIDIA CUDA technolgy for optimization.
Tested on Qt 5.11.2 and CUDA 10.0 Toolkit (Windows 8.1 and Linux Mint 19 Cinammon)

### Building on Windows
1. Create Visual Studio CMake project from repository files
2. In Visual Studio top menu choose CMake > Build

### Building on Linux
1. Clone repository
2. `cd <repo_path>`
3. `cmake . && make`

Executables are generated in `build/bin` directory

### Testing
Test output NVIDIA GEFORCE GTX 1080 Ti :
```
2.5DPaint/build $ ./bin/2.5DPaint --test 2> /dev/null
Default brush tests:
[CPU] init/resize cpu buffers (1234, 1234)
[GPU] init/resize GPU buffers (1234, 1234)
Average time for CPU: 22.1428ms
Min time for CPU: 6.16202ms
Average time for GPU: 0.686615ms
Min time for GPU: 0.429499ms
Average error: 5.45064e-05
Max error in pixel space (int from [0,255]): 1

Textured brush tests:
[CPU] init/resize cpu buffers (1234, 1234)
[GPU] init/resize GPU buffers (1234, 1234)
Average time for CPU: 27.1218ms
Min time for CPU: 7.36051ms
Average time for GPU: 0.734355ms
Min time for GPU: 0.458586ms
Average error: 3.15218e-05
Max error in pixel space (int from [0,255]): 1
```
