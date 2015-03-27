// ###
// ###
// ### Practical Course: GPU Programming in Computer Vision
// ### Final Project: Variational Depth from Focus
// ###
// ### Technical University Munich, Computer Vision Group
// ### Summer Semester 2014, September 8 - October 10
// ###
// ###
// ### Maria Klodt, Jan Stuehmer, Mohamed Souiai, Thomas Moellenhoff
// ###
// ###

// ### Dennis Mack, dennis.mack@tum.de, p060
// ### Adrian Haarbach, haarbach@in.tum.de, p077
// ### Markus Schlaffer, markus.schlaffer@in.tum.de, p070

#ifndef CUDA_WRAPPERS_H
#define CUDA_WRAPPERS_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <utils.cuh>
#include <DataPreparator.cuh>
#include <common_kernels.cuh>

#include <cublas_v2.h>

namespace vdff {
#define CUDA_NAIVE_IMPL
  //#define CUDA_CONVOLUTION_GLOBAL
#define CUDA_CONVOLUTION_SHARED

  extern void cudaSubtractArrays(dim3 gridSize, dim3 blockSize,
				 float *d_a, float *d_b, float *d_out, int w, int h, int nc, cublasHandle_t handle=0);

  extern void cudaAddArrays(dim3 gridSize, dim3 blockSize,
			    float *d_a, float *d_b, float *d_out, int w, int h, int nc, cublasHandle_t handle=0);

  extern void cudaCalcBackwardDifferences(dim3 gridSize, dim3 blockSize,
					  float *imgIn, float *v1, float *v2, int w, int h, int nc,
					  BoundaryBehavior behavior=REPLICATE);

  extern void cudaCalcBackwardDifferencesXDirection(dim3 gridSize, dim3 blockSize,
						    float *imgIn, float *dx, int w, int h, int nc,
						    BoundaryBehavior behavior=REPLICATE);

  extern void cudaCalcBackwardDifferencesYDirection(dim3 gridSize, dim3 blockSize,
						    float *imgIn, float *dy, int w, int h, int nc,
						    BoundaryBehavior behavior=REPLICATE);

  extern void cudaCalcForwardDifferences(dim3 gridSize, dim3 blockSize,
					 float *imgIn, float *v1, float *v2, int w, int h, int nc,
					 BoundaryBehavior behavior=REPLICATE);

  extern void cudaMultiplyArrayWithScalar(dim3 gridSize, dim3 blockSize,
					  float *d_arr, float s, float *d_out, int w, int h, int nc, size_t yOffset=0,
					  cublasHandle_t handle=0);

  extern void cudaConvolution(dim3 gridSize, dim3 blockSize,
			      float *imgIn, float *kernel, float *imgOut, int  w, int h, int nc, int kernelRadius);

  extern void cudaPolyfit(dim3 gridSize, dim3 blockSize,
			  const float *d_pInv, const float *d_sharpness, float *coef, const int w, const int h, const int n, const int degree, size_t yOffset=0);

  extern void cudaPolyder(dim3 gridSize, dim3 blockSize,
			  const float *coef, float *coefDerivative, const int w, const int h, const int degree, size_t yOffset=0);

  /* extern void cudaPolyval(dim3 gridSize, dim3 blockSize, */
			
  extern void cudaComputeMLAP(dim3 gridSize, dim3 blockSize,
			      const float *d_img, float *d_MLAPEstimate, const int w, const int h, const int nc, const int nrImgs=1,
			      const cudaStream_t streamID=0);

  extern void cudaPolyval(dim3 gridSize, dim3 blockSize,
			  const float *d_coefImg, const float *d_x0Img, float *y0Img, const int w, const int h, const int nc);
			
  extern void cudaFindMax(dim3 gridSize, dim3 blockSize,
			  const float *values, float *maxValues, float *indicesMaxValues, const int w, const int h, const int nrImgs);

  extern void cudaScaleSharpnessValues(dim3 gridSize, dim3 blockSize,
				       const float *maxValues, float *sharpness, const int w, const int h, const int nrImgs, const float denomRegu = 0.1f);
}
#endif
