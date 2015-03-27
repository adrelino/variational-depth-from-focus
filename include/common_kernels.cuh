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
#ifndef COMMON_KERNELS_CUH
#define COMMON_KERNELS_CUH

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Header file for our common used kernels
namespace vdff {
  typedef enum {
    REPLICATE,
    ZERO,
    SYMMETRIC
  }BoundaryBehavior;

  __global__ void createLaplacianKernelGrid(float *grid, int *sizes, int *repeats, int *hvec, int nrGridElements);

  __global__ void createLaplacianKernel(float *grid, float *kernel, int nrDimensions, int nrGridElements);

  __device__ void calcBoundedIndices(float *xIdx, float *yIdx, BoundaryBehavior behavior, int w, int h);

  __global__ void calcForwardDifferences(float *imgIn, float *v1, float *v2, int w, int h, int nc,
					 BoundaryBehavior behavior=REPLICATE);

  //for matlab code
  __global__ void calcForwardDifferencesOld(const float *imgIn, float *dx, float *dy, const int w, const int h, const int nc);
  __global__ void calcSecondDerivativeOld(const float *imgIn, float *dxx, float *dyy, const int w, const int h, const int nc);


  __global__ void calcBackwardDifferences(float *imgIn, float *v1, float *v2, int w, int h, int nc,
					  BoundaryBehavior behavior=REPLICATE);

  __global__ void calcBackwardDifferencesXDirection(float *imgIn, float *dx, int w, int h, int nc,
						    BoundaryBehavior behavior=REPLICATE);

  __global__ void calcBackwardDifferencesYDirection(float *imgIn, float *dy, int w, int h, int nc,
						    BoundaryBehavior behavior=REPLICATE);

  __global__ void isoShrinkage(float *d_Gx, float *d_Gy, float *d_GxOut, float *d_GyOut,
			       float sigma, int w, int h, int nc);

  __global__ void addArrays(float *d_a, float *d_b, float *d_out, int w, int h, int nc);

  __global__ void subtractArrays(float *d_a, float *d_b, float *d_out, int w, int h, int nc);

  __global__ void divideArrays(float *d_a, float *d_b, float *d_out, int w, int h, int nc);

  __global__ void multiplyArrayWithScalar(float *d_arr, float s, float *d_out, int w, int h, int nc, size_t offset=0);

  __global__ void addScalarToArray(float *d_arr, float s, float *d_out, int w, int h, int nc);

  __global__ void thresholdArray(float *d_arr, float threshLow, float threshHigh, float *d_out, int w, int h, int nc);

  __global__ void convolutionShared(float *imgIn, float *kernel, float *imgOut,
				    int w, int h, int nc, int kernelRadius, int shw, int shh);

  __global__ void convolutionGlobal(float *imgIn, float *kernel, float *imgOut, int w, int h, int nc, int kernelSize);

  __global__ void MLAP(const float *imgIn, float *imgOut, const int w, const int h, const int nc, const int nrImgs=1);

  __host__ __device__ void matrixVectorMul(const float *Xpinv, const float *yVec,  float *coefVec, const int n, const int m, const int layer_size=1);

  __host__ __device__ void matrixVectorMulChunked(const float *Xpinv, const float *yVec,  float *coefVec, const int n, const int m,
						  const size_t layerSizeYVec=1, const size_t layerSizeCoefVec=1);

  __global__ void matrixPolyfit(const float *Xpinv, const float *sharpness, float *coefImg, const int w, const int h, const int n, const int m, size_t offset=0);

  __global__ void matrixPolyfitChunked(const float *Xpinv, const float *sharpness, float *coefImg,
				       const int wSharpness, const int hSharpness, const int n,
				       const int wCoefImg, const int hCoefImg, const int m, size_t yOffsetCoefImg);

  __global__ void matrixPolyfitNewLayout(const float *Xpinv, const float *sharpness, float *coefImg, const int w, const int h, const int n, const int m, size_t offset=0);

  __global__ void matrixPolyder(const float *coefImg, float *coefImgDer, const int w, const int h, const int m);

  __global__ void matrixPolyderNewLayout(const float *coefImg, float *coefImgDer, const int w, const int h, const int m, size_t offset);

  __host__ __device__ float polyval(const float *coefVec, const float x0, const int nc, const int layer_size=1);

  __global__ void matrixPolyval(const float *coefImg, const float *x0Img, float *y0Img, const int w, const int h, const int nc);

  __global__ void matrixPolyvalNewLayout(const float *coefImg, const float *x0Img, float *y0Img, const int w, const int h, const int nc);

  __global__ void findMax(const float *values, float *maxValues, float *indicesMaxValues, const int w, const int h, const int nc);

  __global__ void findMaxIndices(const float *values, float *indicesMaxValues, const int wValues, const int hValues, const int nc,
				 const size_t wIndices, const size_t hIndices, const size_t yOffset);

  __global__ void scaleSharpnessValuesGPU(const float *maxValues, float *sharpness, const int w, const int h, const int nc, const float denomRegu = 0.1f);

  //for fct
  __global__ void cuFCTinitIndex(size_t *index, int w, int h);

  __global__ void cuFCTinitRoots(float2 *roots, int w, int h);

  __global__ void cuFCTCalcOutput(float2 *v, float *output, float2 *roots, int w, int h);

  __global__ void cuIFCTPrepareInput(float *input, float2 *v, float2 *roots, int w, int h);

  //resorts array, so output[i]=input[v_index[i]]
  __global__ void resort2DArrayForward(float *input, float *output, size_t *v_index, int w, int h);
  //resorts array, so output[v_index[i]]=input[i]
  __global__ void resort2DArrayBackward(float *input, float *output, size_t *v_index, int w, int h);
}

#endif //COMMON_KERNELS_CUH
