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

#include <cudaWrappers.h>

#include <cublas_v2.h>

void cudaSubtractArrays(dim3 gridSize, dim3 blockSize,
			float *d_a, float *d_b, float *d_out, int w, int h, int nc, cublasHandle_t handle) {
  #ifdef CUDA_NAIVE_IMPL
  subtractArrays<<<gridSize, blockSize>>>(d_a, d_b, d_out, w, h, nc); CUDA_CHECK;
  #elif defined CUDA_CUBLAS
  cublasSaxpy(handle, w*h*nc, -1.0f, d_a, 1, d_b, 1);
  d_out = d_b;
  #endif
}

void cudaAddArrays(dim3 gridSize, dim3 blockSize,
		   float *d_a, float *d_b, float *d_out, int w, int h, int nc, cublasHandle_t handle) {
  #ifdef CUDA_NAIVE_IMPL
  addArrays<<<gridSize, blockSize>>>(d_a, d_b, d_out, w, h, nc); CUDA_CHECK;
  #elif defined CUDA_CUBLAS
  cublasSaxpy(handle, w*h*nc, 1.0f, d_a, 1, d_b, 1);
  d_out = d_b;  
  #endif
}

void cudaCalcBackwardDifferences(dim3 gridSize, dim3 blockSize,
				 float *imgIn, float *v1, float *v2, int w, int h, int nc,
				 BoundaryBehavior behavior) {
  #ifdef CUDA_NAIVE_IMPL
  calcBackwardDifferences<<<gridSize, blockSize>>>(imgIn, v1, v2, w, h, nc, behavior);
  #endif  
}

void cudaCalcBackwardDifferencesXDirection(dim3 gridSize, dim3 blockSize,
  float *imgIn, float *dx, int w, int h, int nc,BoundaryBehavior behavior) {
  #ifdef CUDA_NAIVE_IMPL
  calcBackwardDifferencesXDirection<<<gridSize, blockSize>>>(imgIn, dx, w, h, nc, behavior);
  #endif    
}

void cudaCalcBackwardDifferencesYDirection(dim3 gridSize, dim3 blockSize,
	float *imgIn, float *dy, int w, int h, int nc,BoundaryBehavior behavior) {
  #ifdef CUDA_NAIVE_IMPL
  calcBackwardDifferencesYDirection<<<gridSize, blockSize>>>(imgIn, dy, w, h, nc, behavior);
  #endif    
}

void cudaCalcForwardDifferences(dim3 gridSize, dim3 blockSize,
	float *imgIn, float *v1, float *v2, int w, int h, int nc,BoundaryBehavior behavior) {
  #ifdef CUDA_NAIVE_IMPL
  calcForwardDifferences<<<gridSize, blockSize>>>(imgIn, v1, v2, w, h, nc, behavior);
  #endif  
}

void cudaMultiplyArrayWithScalar(dim3 gridSize, dim3 blockSize,
	float *d_arr, float s, float *d_out, int w, int h, int nc, size_t yOffset, cublasHandle_t handle) {
  #ifdef CUDA_NAIVE_IMPL
  multiplyArrayWithScalar<<<gridSize, blockSize>>>(d_arr, s, d_out, w, h, nc, yOffset); CUDA_CHECK;
  #elif defined CUDA_CUBLAS
  cublasSscal(handle, w*h*nc, s, d_arr, 1);
  d_arr = d_out;  
  #endif    
}

void cudaConvolution(dim3 gridSize, dim3 blockSize, 
	float *imgIn, float *kernel, float *imgOut, int  w, int h, int nc, int kernelRadius) {
  #if defined CUDA_CONVOLUTION_SHARED
  size_t shw = blockSize.x + 2*kernelRadius;
  size_t shh = blockSize.y + 2*kernelRadius;  
  size_t smBytes = shw * shh * sizeof(float);
  
  convolutionShared<<<gridSize, blockSize, smBytes>>>(imgIn, kernel, imgOut, w, h, nc, kernelRadius, shw, shh); CUDA_CHECK;
  #elif defined CUDA_CONVOLUTION_GLOBAL
  convolutionGlobal<<<gridSize, blockSize>>>(imgIn, kernel, imgOut, w, h, nc, 2*kernelRadius + 1);
  #endif
}

void cudaPolyfit(dim3 gridSize, dim3 blockSize,
	const float *d_pInv, const float *d_sharpness, float *coef, const int w, const int h, const int n, const int degree, size_t yOffset) {
  matrixPolyfit<<<gridSize, blockSize>>>(d_pInv, d_sharpness, coef, w, h, n, degree, yOffset);CUDA_CHECK;
}

void cudaPolyder(dim3 gridSize, dim3 blockSize,
	const float *coef, float *coefDerivative, const int w, const int h, const int degree,size_t yOffset) {
  matrixPolyder<<<gridSize, blockSize>>>(coef, coefDerivative, w, h, degree);CUDA_CHECK;
}

void cudaPolyval(dim3 gridSize, dim3 blockSize,
	const float *d_coefImg, const float *d_x0Img, float *d_y0Img, const int w, const int h, const int nc) {
  matrixPolyval<<<gridSize, blockSize>>>(d_coefImg, d_x0Img, d_y0Img, w, h, nc);CUDA_CHECK;        
}

// TODO: actually use reduction: http://www.cuvilib.com/Reduction.pdf!!
void cudaFindMax(dim3 gridSize, dim3 blockSize,
  const float *values, float *maxValues, float *indicesMaxValues, const int w, const int h, const int nrImgs) {
  findMax<<<gridSize, blockSize>>>(values, maxValues, indicesMaxValues, w, h, nrImgs);CUDA_CHECK;
}

void cudaScaleSharpnessValues(dim3 gridSize, dim3 blockSize,
  const float *maxValues, float *sharpness, const int w, const int h, const int nrImgs, const float denomRegu){
  scaleSharpnessValuesGPU<<<gridSize, blockSize>>>(maxValues,sharpness,w,h,nrImgs,denomRegu);CUDA_CHECK;
}

void cudaComputeMLAP(dim3 gridSize, dim3 blockSize,
	const float *d_img, float *d_MLAPEstimate, const int w, const int h, const int nc, const int nrImgs,
	const cudaStream_t streamID) {
  MLAP<<<gridSize, blockSize, 0, streamID>>>(d_img, d_MLAPEstimate, w, h, nc, nrImgs); CUDA_CHECK;
}

