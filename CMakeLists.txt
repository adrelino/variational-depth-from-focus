cmake_minimum_required(VERSION 2.8)

project(variational-depth-from-focus)

# find cuda and openCV
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

# specify include directories
include_directories(include/)

# sources
set(SOURCES
    src/main.cu
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
  main
  ${SOURCES}
  src/wintime.cpp
  )    
else()
cuda_add_executable(
  main
  ${SOURCES}
  )  
endif()


cuda_add_cufft_to_target(main)
cuda_add_cublas_to_target(main)

target_link_libraries(
  main
  ${OpenCV_LIBS}
  )
