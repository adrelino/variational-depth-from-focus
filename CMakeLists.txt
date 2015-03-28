cmake_minimum_required(VERSION 2.8)

project(variational-depth-from-focus)

# find cuda and openCV
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

# specify include directories
include_directories(include/)

# headers to show up in qt creator
file(GLOB HEADERS
    include/*.h$
    include/*.hpp$
    include/*.cuh$
)

# sources
set(SOURCES
    src/vdff.cu
    src/common_kernels.cu
    src/LaplaceInversion.cu
    src/FCT.cu
    src/openCVHelpers.cpp
    src/CUDATimer.cpp
    src/cudaWrappers.cu
    src/utils.cu
    src/Tensor3f.cu
    src/LinearizedADMM.cu
    src/DataPreparator.cu
    src/DataPreparatorTensor3f.cu
    src/LayeredMemory.cu
    src/CPUTimer.cpp
)

# this will be the main entry point for the project
if (MSVC)
cuda_add_executable(
  vdff
  ${SOURCES}
  src/wintime.cpp
  )    
else()
cuda_add_executable(
  vdff
  ${SOURCES}
  ${HEADERS}
  )  
endif()


cuda_add_cufft_to_target(vdff)
cuda_add_cublas_to_target(vdff)

target_link_libraries(
  vdff
  ${OpenCV_LIBS}
  )

add_executable(
  compress
  src/compress.cpp
  src/openCVHelpers.cpp
)

target_link_libraries(
  compress
  ${OpenCV_LIBS}
  )


add_executable(
  compare
  src/compare.cpp
  src/openCVHelpers.cpp
)

target_link_libraries(
  compare
  ${OpenCV_LIBS}
  )
