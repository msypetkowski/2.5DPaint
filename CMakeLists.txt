cmake_minimum_required(VERSION 3.1.0)

project(2.5DPaint)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
set(CMAKE_BINARY_DIR "${CMAKE_SOURCE_DIR}/build")

# set output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_CURRENT_BINARY_DIR ${CMAKE_BINARY_DIR})

# QT
# Set your qt version and configuration (example: gcc_64, msvc2015)
set(QT_VERSION "5.11.2")

if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    set(QT_CONFIG "gcc_64")
	set(CMAKE_PREFIX_PATH "~/Qt${QT_VERSION}/${QT_VERSION}/${QT_CONFIG}")
elseif(WIN32 OR WIN64)
    set(QT_CONFIG "msvc2017_64")
    set(CMAKE_PREFIX_PATH "C:\\Qt\\${QT_VERSION}\\${QT_CONFIG}\\")
endif()

# Find includes in corresponding build directories
# set(CMAKE_INCLUDE_CURRENT_DIR ON)
# Instruct CMake to run moc automatically when needed
set(CMAKE_AUTOMOC ON)
# Create code from a list of Qt designer ui files
set(CMAKE_AUTOUIC ON)

# Find the QtWidgets library
find_package(Qt5Widgets CONFIG REQUIRED)
find_package( OpenCV REQUIRED )

# OPENGL
# find_package(OpenGL REQUIRED)

set(QT_PATH ${Qt5Core_DIR}/../../..)

# Populate a CMake variable with the sources
set(SRC_DIR "src")
set(GUI_SRC_DIR "${SRC_DIR}/gui")
set(CUDA_SRC_DIR "${SRC_DIR}/cuda")
include_directories(${GUI_SRC_DIR} ${CUDA_SRC_DIR} "${SRC_DIR}/include")

file(GLOB_RECURSE PROJECT_SRCS
	"${GUI_SRC_DIR}/*.h"
	"${GUI_SRC_DIR}/*.hpp"
    "${GUI_SRC_DIR}/*.cpp"

    "${SRC_DIR}/*.h"
	"${SRC_DIR}/*.hpp"
    "${SRC_DIR}/*.cpp"
    "${SRC_DIR}/*.c"

	"${CUDA_SRC_DIR}/*.cu"
        "${CUDA_SRC_DIR}/*.cpp"
        "${CUDA_SRC_DIR}/*.cuh"
	"${CUDA_SRC_DIR}/*.h"
)

# Enable C++11 for host code
set(CMAKE_CXX_STANDARD 11)

#CUDA

# Enable CUDA debug info in debug mode builds
list(APPEND CUDA_NVCC_FLAGS_DEBUG -G -g)

# Specify CUDA nvcc flags
list(APPEND CUDA_NVCC_FLAGS -arch=sm_35)

# Linux-specific hacks/fixes
if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    list(APPEND CMAKE_EXE_LINKER_FLAGS "-lX11 -lXxf86vm")
endif()

# CUDA linking
find_package(Threads REQUIRED)
find_package(CUDA REQUIRED)

set(CUDA_GENERATED_OUTPUT_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE ON)
set(CUDA_SEPARABLE_COMPILATION OFF)
set(CUDA_VERBOSE_BUILD ON)

# include cuda samples headers
if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    include_directories("~/NVIDIA_CUDA-${CUDA_VERSION_STRING}_Samples/common/inc")
    include_directories("/opt/cuda/samples/common/inc/") # default arch linux cuda inc path
elseif(WIN32 OR WIN64)
    include_directories("$ENV{NVCUDASAMPLES_ROOT}\\common\\inc")
endif()

#executables

# Tell CMake to create the helloworld executable
cuda_add_executable(${CMAKE_PROJECT_NAME} ${PROJECT_SRCS})
# Use the Widgets module from Qt 5

if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    target_link_libraries(${CMAKE_PROJECT_NAME} Qt5::Widgets ${OpenCV_LIBS})
elseif(WIN32 OR WIN64)
    # TODO(Konrad): Propably find a way that won't break everybody's builds
    target_link_libraries(${CMAKE_PROJECT_NAME} Qt5::Widgets ${OpenCV_LIBS} "C:\\Users\\Memotosz\\Libs\\opencv\\build\\x64\\vc15\\lib\\opencv_world343.lib")
endif()

# additional actions
if(WIN32 OR WIN64)
	add_custom_target(Qt5CopyBinaries
		COMMAND ${CMAKE_COMMAND} -E copy_if_different ${QT_PATH}/bin/Qt5Core.dll ${CMAKE_BINARY_DIR}/bin
		COMMAND ${CMAKE_COMMAND} -E copy_if_different ${QT_PATH}/bin/Qt5Widgets.dll ${CMAKE_BINARY_DIR}/bin
		COMMAND ${CMAKE_COMMAND} -E copy_if_different ${QT_PATH}/bin/Qt5Gui.dll ${CMAKE_BINARY_DIR}/bin
		#debug
		COMMAND ${CMAKE_COMMAND} -E copy_if_different ${QT_PATH}/bin/Qt5Cored.dll ${CMAKE_BINARY_DIR}/bin
		COMMAND ${CMAKE_COMMAND} -E copy_if_different ${QT_PATH}/bin/Qt5Widgetsd.dll ${CMAKE_BINARY_DIR}/bin
		COMMAND ${CMAKE_COMMAND} -E copy_if_different ${QT_PATH}/bin/Qt5Guid.dll ${CMAKE_BINARY_DIR}/bin
	)
	add_dependencies(${CMAKE_PROJECT_NAME} Qt5CopyBinaries)
endif()

add_custom_target(CopyShaderFiles
	COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/shaders
	COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_SOURCE_DIR}/shaders/default.vert ${CMAKE_BINARY_DIR}/bin/shaders
	COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_SOURCE_DIR}/shaders/default.frag ${CMAKE_BINARY_DIR}/bin/shaders
)
add_dependencies(${CMAKE_PROJECT_NAME} CopyShaderFiles)
