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

#include <common_kernels.cuh>
#include <math_constants.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>
#include <stdio.h>
#include "id.h"
#include <cuComplex.h>

namespace vdff {
  // for both the laplacian kernels we are assume that the blocks are 1D, and the grid is 2D
  // where x = (nrGridElements / blockSize.x) and y = nrDimensions
  __global__ void createLaplacianKernelGrid(float *grid, int *sizes, int *repeats, int *hvec,
					    int nrGridElements) {
    size_t x = threadIdx.x + blockDim.x * blockIdx.x;
    size_t dim = blockIdx.y;

    if (x >= nrGridElements)
      return;

    int curSize = sizes[dim];
    int denom = hvec[dim];

    int gridIdx = x / repeats[dim];
    gridIdx %= curSize;
    float val = sinf((CUDART_PI_F * gridIdx) / (2.0f * curSize)) / denom;
    grid[x + dim*nrGridElements] = 4*(val*val);
  }

  __global__ void createLaplacianKernel(float *grid, float *kernel, int nrDimensions, int nrGridElements) {
    size_t x = threadIdx.x + blockDim.x * blockIdx.x;

    if (x >= nrGridElements)
      return;

    for(int d = 0; d < nrDimensions; ++d) {
      if (d == 0)
	kernel[x] = grid[x];
      else
	kernel[x] += grid[x + d*nrGridElements];
    }
  }

  __device__ void boundaryBehavior(int *xIdx, int *yIdx, BoundaryBehavior behavior, int w, int h) {
    switch(behavior) {
    case REPLICATE:
      *xIdx = max(min(*xIdx, w-1), 0);
      *yIdx = max(min(*yIdx, h-1), 0);    
      break;
    
    case ZERO:
      if (*xIdx > w-1 || *xIdx < 0)
	*xIdx = -1;
    
      if (*yIdx > h-1 || *yIdx < 0)
	*yIdx = -1;
      break;
    
    case SYMMETRIC:
      //TODO(Dennis)
      break;

    default:
      break;
    }
  }

  __global__ void calcForwardDifferences(float *imgIn, float *dx, float *dy, int w, int h, int nc,
					 BoundaryBehavior behavior){
    size_t x = threadIdx.x + blockDim.x * blockIdx.x;
    size_t y = threadIdx.y + blockDim.y * blockIdx.y;

    if(x >= w || y >= h) return;

    // boundary behavior: currently 'replicate'
    int xPlus = x + 1;
    int yPlus = y + 1;
    
    boundaryBehavior(&xPlus, &yPlus, behavior, w, h);

    for (int c = 0; c < nc; ++c)
      {
	dx[id(x,y,w,h,c)] = ((xPlus == -1) ? 0.0f : imgIn[id(xPlus,y,w,h,c)]) - imgIn[id(x,y,w,h,c)];
	dy[id(x,y,w,h,c)] = ((yPlus == -1) ? 0.0f : imgIn[id(x,yPlus,w,h,c)]) - imgIn[id(x,y,w,h,c)];
      }
  }

  __global__ void calcForwardDifferencesOld(const float *imgIn, float *dx, float *dy, const int w, const int h, const int nc){
    size_t x = threadIdx.x + blockDim.x * blockIdx.x;
    size_t y = threadIdx.y + blockDim.y * blockIdx.y;

    if(x >= w || y >= h) return;

    // boundary behavior: currently 'replicate'
    int xPlus = x + 1;
    if(xPlus >= w) xPlus = w-1;

    int yPlus = y + 1;
    if(yPlus >= h) yPlus = h-1;

    for (int l = 0; l < nc; ++l){
      dx[id(x,y,w,h,l)] = imgIn[id(xPlus,y,w,h,l)] - imgIn[id(x,y,w,h,l)];
      dy[id(x,y,w,h,l)] = imgIn[id(x,yPlus,w,h,l)] - imgIn[id(x,y,w,h,l)];
    }
  }

  __global__ void calcSecondDerivativeOld(const float *imgIn, float *dxx, float *dyy, const int w, const int h, const int nc){
    size_t x = threadIdx.x + blockDim.x * blockIdx.x;
    size_t y = threadIdx.y + blockDim.y * blockIdx.y;

    if(x >= w || y >= h) return;

    // boundary behavior: currently 'replicate'
    int xPlus1 = x+1;
    int xMinus1 = x-1;

    int yPlus1 = y+1;
    int yMinus1 = y-1;

    // do clamping
    xPlus1 = max(min(xPlus1, w-1), 0);
    xMinus1 = min(max(xMinus1, 0), w-1);

    yPlus1 = max(min(yPlus1, h-1), 0);
    yMinus1 = min(max(yMinus1, 0), h-1);

    for (int l = 0; l < nc; ++l){
      dxx[id(x,y,w,h,l)] = imgIn[id(xMinus1,y,w,h,l)] - 2*imgIn[id(x,y,w,h,l)] + imgIn[id(xPlus1,y,w,h,l)];
      dyy[id(x,y,w,h,l)] = imgIn[id(x,yMinus1,w,h,l)] - 2*imgIn[id(x,y,w,h,l)] + imgIn[id(x,yPlus1,w,h,l)];
    }
  }

  __global__ void calcBackwardDifferences(float *imgIn, float *dx, float *dy, int w, int h, int nc,
					  BoundaryBehavior behavior){
    size_t x = threadIdx.x + blockDim.x * blockIdx.x;
    size_t y = threadIdx.y + blockDim.y * blockIdx.y;

    if(x >= w || y >= h) return;

    // boundary behavior: currently 'replicate'
    int xMinus = x - 1;
    int yMinus = y - 1;

    boundaryBehavior(&xMinus, &yMinus, behavior, w, h);
    
    for (int i = 0; i < nc; ++i)
      {
	dx[id(x,y,w,h,i)] = imgIn[id(x,y,w,h,i)] - ((xMinus == -1) ? 0.0f : imgIn[id(xMinus,y,w,h,i)]);
	dy[id(x,y,w,h,i)] = imgIn[id(x,y,w,h,i)] - ((yMinus == -1) ? 0.0f : imgIn[id(x,yMinus,w,h,i)]);      
      }
  }

  __global__ void calcBackwardDifferencesXDirection(float *imgIn, float *dx, int w, int h, int nc,
						    BoundaryBehavior behavior) {
    size_t x = threadIdx.x + blockDim.x * blockIdx.x;
    size_t y = threadIdx.y + blockDim.y * blockIdx.y;

    if(x >= w || y >= h) return;

    // boundary behavior: currently 'replicate'
    int xMinus = x - 1;
    int yMinus = y - 1;

    boundaryBehavior(&xMinus, &yMinus, behavior, w, h);
    
    for (int i = 0; i < nc; ++i)
      {
	dx[id(x,y,w,h,i)] = imgIn[id(x,y,w,h,i)] - ((xMinus == -1) ? 0.0f : imgIn[id(xMinus,y,w,h,i)]);
      }  
  }

  __global__ void calcBackwardDifferencesYDirection(float *imgIn, float *dy, int w, int h, int nc,
						    BoundaryBehavior behavior) {
    size_t x = threadIdx.x + blockDim.x * blockIdx.x;
    size_t y = threadIdx.y + blockDim.y * blockIdx.y;

    if(x >= w || y >= h) return;

    // boundary behavior: currently 'replicate'
    int xMinus = x - 1;
    int yMinus = y - 1;

    boundaryBehavior(&xMinus, &yMinus, behavior, w, h);
    
    for (int i = 0; i < nc; ++i)
      {
	dy[id(x,y,w,h,i)] = imgIn[id(x,y,w,h,i)] - ((yMinus == -1) ? 0.0f : imgIn[id(x,yMinus,w,h,i)]);      
      }  
  }

  // isoShrinkage function as defined in paper (or at least similar) 
  __global__ void isoShrinkage(float *d_Gx, float *d_Gy, float *d_GxOut, float *d_GyOut,
			       float sigma, int w, int h, int nc) {
    size_t x = threadIdx.x + blockDim.x * blockIdx.x;
    size_t y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x >= w || y >= h)
      return;

    for (int c = 0; c < nc; ++c) {
      float valGx = d_Gx[x + y*w + c*w*h];
      float valGy = d_Gy[x + y*w + c*w*h];

      float mag = sqrtf(valGx*valGx + valGy*valGy);
      float normBoundAway = fmaxf(mag, sigma / 2.0f);
      float maxVal = fmaxf(mag - sigma, 0);

      d_GxOut[id(x,y,w,h,c)] = (maxVal / normBoundAway) * valGx;
      d_GyOut[id(x,y,w,h,c)] = (maxVal / normBoundAway) * valGy;
    }
  }

  __global__ void addArrays(float *d_a, float *d_b, float *d_out, int w, int h, int nc) {
    size_t x = threadIdx.x + blockDim.x * blockIdx.x;
    size_t y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x >= w || y >= h)
      return;

    for (int c = 0; c < nc; ++c) {
      d_out[id(x,y,w,h,c)] = d_a[id(x,y,w,h,c)] + d_b[id(x,y,w,h,c)];
    }
  }

  __global__ void subtractArrays(float *d_a, float *d_b, float *d_out, int w, int h, int nc) {
    size_t x = threadIdx.x + blockDim.x * blockIdx.x;
    size_t y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x >= w || y >= h)
      return;

    for (int c = 0; c < nc; ++c) {
      d_out[id(x,y,w,h,c)] = d_a[id(x,y,w,h,c)] - d_b[id(x,y,w,h,c)];    
    }
  }

  __global__ void divideArrays(float *d_a, float *d_b, float *d_out, int w, int h, int nc) {
    size_t x = threadIdx.x + blockDim.x * blockIdx.x;
    size_t y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x >= w || y >= h)
      return;

    for (int c = 0; c < nc; ++c) {
      d_out[id(x,y,w,h,c)] = d_a[id(x,y,w,h,c)] / d_b[id(x,y,w,h,c)];    
    }
  }

  __global__ void multiplyArrayWithScalar(float *d_arr, float s, float *d_out, int w, int h, int nc, size_t yOffset) {
    size_t x = threadIdx.x + blockDim.x * blockIdx.x;
    size_t y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x >= w || y >= h)
      return;

    for(int c = 0; c < nc; ++c)
      d_out[id(x,y,w,h,c)] = s * d_arr[id(x,y,w,h,c)];
  }

  __global__ void addScalarToArray(float *d_arr, float s, float *d_out, int w, int h, int nc) {
    size_t x = threadIdx.x + blockDim.x * blockIdx.x;
    size_t y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x >= w || y >= h)
      return;

    for(int c = 0; c < nc; ++c)
      d_out[id(x,y,w,h,c)] = s + d_arr[id(x,y,w,h,c)];
  }

  __global__ void thresholdArray(float *d_arr, float threshLow, float threshHigh, float *d_out, int w, int h, int nc) {
    size_t x = threadIdx.x + blockDim.x * blockIdx.x;
    size_t y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x >= w || y >= h)
      return;

    for (int c = 0; c < nc; ++c) {
      float val = d_arr[x + y*w + c*w*h];

      if (val > threshHigh)
	val = threshHigh;
      else if(val < threshLow)
	val = threshLow;

      d_out[id(x,y,w,h,c)] = val;
    }
  }

  // assumes that the kernel is in shared memory
  __global__ void convolutionShared(float *imgIn, float *kernel, float *imgOut,
				    int w, int h, int nc, int kernelRadius, int shw, int shh){
    size_t x = threadIdx.x + blockDim.x * blockIdx.x;
    size_t y = threadIdx.y + blockDim.y * blockIdx.y;

    extern __shared__ float shmem[];

    int r = kernelRadius;
    int kernelSize = 2*r + 1;

    int tx=threadIdx.x;
    int bx=blockIdx.x;
    int ty=threadIdx.y;
    int by=blockIdx.y;   
    int bdx=blockDim.x;
    int bdy=blockDim.y;   

    for(unsigned int c=0;c<nc;c++,__syncthreads()) {

      ////////
      //step 1: copy data into shared memory, with clamping padding
      //
      for(int pt=tx+bdx*ty ; pt<shw*shh ;pt+=bdx*bdy){
	int xi = (pt % shw) + (bx *bdx - r);
	int yi = (pt / shw) + (by *bdy - r);

	xi = max(min(xi,w-1),0);
	yi = max(min(yi,h-1),0);

	float val=imgIn[id(xi,yi,w,h,c)];
	shmem[pt] = val; 
      }

      __syncthreads();


      ///////
      //step 2: convolution, no more clamping needed
      //
      if(x>=w || y>=h) continue; //check for block border only AFTER copying to shared mem (goes over block borders)

      float sum=0;

      //convolution using adrian + markus indexing
      for(int i=0;i<kernelSize;i++){
	for(int j=0;j<kernelSize;j++){
	  int x_new=threadIdx.x+i;
	  int y_new=threadIdx.y+j;
	  // no need for id since both are row vectors
	  sum+=kernel[i+j*kernelSize]*shmem[x_new+y_new*shw];
	}
      }
      imgOut[id(x,y,w,h,c)]=sum;
    }
  }

  __global__ void convolutionGlobal(float *imgIn, float *kernel, float *imgOut, int w, int h, int nc, int kernelSize){
    size_t x = threadIdx.x + blockDim.x * blockIdx.x;
    size_t y = threadIdx.y + blockDim.y * blockIdx.y;
    size_t k = kernelSize;

    int r=k/2;

    //check for boundarys of the block
    if(x>=w || y>=h) return; 

    //iterate over all channels
    for(unsigned int c=0;c<nc;c++) {
      float sum=0;
      //do convolution
      for(unsigned int i=0;i<k;i++){
	unsigned int x_new;
	//clamping x
	if(x+r<i) x_new=0;
	else if(x+r-i>=w) x_new=w-1;
	else x_new=x+r-i;
	for(unsigned int j=0;j<k;j++){
	  //clamping y
	  unsigned int y_new;
	  if(y+r<j)
	    y_new=0;
	  else if(y+r-j>=h)
	    y_new=h-1;
	  else
	    y_new=y+r-j;
	  sum+=kernel[i+j*k]*imgIn[x_new+y_new*w+w*h*c];
	}
      }
      //imgOut[id(x,y,w,h,c)]=sum;
      imgOut[x + y*w + c*w*h]=sum;    
    }
  }

  __global__ void MLAP(const float *imgIn, float *imgOut, const int w, const int h, const int nc, const int nrImgs) {
    size_t x = threadIdx.x + blockDim.x * blockIdx.x;
    size_t y = threadIdx.y + blockDim.y * blockIdx.y;

    if(x >= w || y >= h) return;

    // boundary behavior: currently 'replicate'
    int xPlus1 = x+1;
    int xMinus1 = x-1;

    int yPlus1 = y+1;
    int yMinus1 = y-1;

    // do clamping
    xPlus1 = max(min(xPlus1, w-1), 0);
    xMinus1 = min(max(xMinus1, 0), w-1);

    yPlus1 = max(min(yPlus1, h-1), 0);
    yMinus1 = min(max(yMinus1, 0), h-1);

    for(int i=0; i<nrImgs;i++){ //todo: use z index, independent iterations
      float sum=0;
      for (int l = 0; l < nc; ++l) //nc: 1 or 3
	{
	  int channel=l+i*nc;
	  float dxx = imgIn[id(xMinus1,y,w,h,channel)] - 2*imgIn[id(x,y,w,h,channel)] + imgIn[id(xPlus1,y,w,h,channel)];
	  float dyy = imgIn[id(x,yMinus1,w,h,channel)] - 2*imgIn[id(x,y,w,h,channel)] + imgIn[id(x,yPlus1,w,h,channel)];
	  sum+=abs(dxx)+abs(dyy);
	}
      imgOut[id(x,y,w,h,i)]=sum; //grayscale, so channel means nrImgs index
    }
  }

  __global__ void matrixPolyfit(const float *Xpinv, const float *sharpness, float *coefImg, const int w, const int h, const int n, const int m, size_t yOffset){
    size_t x = threadIdx.x + blockDim.x * blockIdx.x;
    size_t y = threadIdx.y + blockDim.y * blockIdx.y;
    
    if(x >= w || y >= h) return;
  
    int layer_size = w*h; // to get from pixel_x,y,n to pixel_x,y,n+1, we need to advance one channel = w*h*sizeof(float) bytes
    // float* yVec = (float*) sharpness + id(x,y,w,h,0); // Matlab: yVec = squeeze(sharpness(i,j,:));
    const float* yVec = &sharpness[id(x,y,w,h,0)];   
    float* coefVec = &coefImg[id(x,y+yOffset,w,h,0)];
  
    matrixVectorMul(Xpinv, yVec, coefVec, n, m, layer_size);
  }

  __global__ void matrixPolyfitChunked(const float *Xpinv, const float *sharpness, float *coefImg,
				       const int wSharpness, const int hSharpness, const int n,
				       const int wCoefImg, const int hCoefImg, const int m, size_t yOffsetCoefImg) {
    size_t x = threadIdx.x + blockDim.x * blockIdx.x;
    size_t y = threadIdx.y + blockDim.y * blockIdx.y;
    
    if(x >= wSharpness || y >= hSharpness) return;

    size_t indexCoefVec = x + y*wCoefImg + yOffsetCoefImg*wCoefImg;
    if (indexCoefVec >= wCoefImg*hCoefImg)
      return;

    size_t layerSizeYVec = wSharpness*hSharpness;
    const float* yVec = &sharpness[id(x,y,wSharpness,hSharpness,0)];

    size_t layerSizeCoefVec = wCoefImg*hCoefImg;
    float* coefVec = &coefImg[indexCoefVec];  
    matrixVectorMulChunked(Xpinv, yVec, coefVec, n, m, layerSizeYVec, layerSizeCoefVec);
  }

  __global__ void matrixPolyfitNewLayout(const float *Xpinv, const float *sharpness, float *coefImg, const int w, const int h, const int n, const int m, size_t indexYOffset){
    size_t x = threadIdx.x + blockDim.x * blockIdx.x;
    size_t y = threadIdx.y + blockDim.y * blockIdx.y;
    
    if(x >= w || y >= h) return;

    // to get from pixel_x,y,n to pixel_x,y,n+1, we need to advance one channel => using new layout this is just the next position
    int layer_size = 1; 

    // float* yVec = (float*) sharpness + id(x,y,w,h,0); // Matlab: yVec = squeeze(sharpness(i,j,:));
    size_t xOffset = n;
    size_t yOffset = w*xOffset;
  
    const float* yVec = &sharpness[x*xOffset + y*yOffset];   

    size_t xOffsetCoef = m;
    size_t yOffsetCoef = w*xOffsetCoef;

    float* coefVec = &coefImg[x*xOffsetCoef + y*yOffsetCoef];
  
    matrixVectorMul(Xpinv, yVec, coefVec, n, m, layer_size);
  }

  //rows x cols					         m x n 				 n x 1 or h x w x n  m x 1 or h x w x m					       is 1 if vectors used, else h x w =l ayer_size
  __host__ __device__ void matrixVectorMul(const float *Xpinv, const float *yVec,  float *coefVec, const int n, const int m, const int layer_size){
    for (int i = 0; i < m; ++i)
      {
	float coef_i=0;

	for (int j = 0; j < n; ++j)
	  {
	    coef_i += Xpinv[id(j,i,n,m,0)] * yVec[j*layer_size]; //layer size used as Stride in yVec
	  }

	coefVec[i*layer_size]=coef_i; //as well as coefVec, since we assume the same layer dimensions in underlying shaprness( h x w x n) or coefImg( h x w x m), even though the third dimension is different
      }
  }

  __host__ __device__ void matrixVectorMulChunked(const float *Xpinv, const float *yVec,  float *coefVec, const int n, const int m,
						  const size_t layerSizeYVec, const size_t layerSizeCoefVec){
    for (int i = 0; i < m; ++i)
      {
	float coef_i=0;

	for (int j = 0; j < n; ++j)
	  {
	    coef_i += Xpinv[id(j,i,n,m,0)] * yVec[j*layerSizeYVec]; //layer size used as Stride in yVec
	  }

	coefVec[i*layerSizeCoefVec]=coef_i;
      }
  }

  __global__ void matrixPolyder(const float *coefImg, float *coefImgDer, const int w, const int h, const int m){
    size_t x = threadIdx.x + blockDim.x * blockIdx.x;
    size_t y = threadIdx.y + blockDim.y * blockIdx.y;
    if(x >= w || y >= h) return;

    for (int i = 0; i < m - 1; ++i) //if of degree d=2, we have n=3 coeffs ax'2 + bx +c
      {
	int x_y_m_idx=id(x,y,w,h,i);
	coefImgDer[x_y_m_idx]=coefImg[x_y_m_idx]*(m-i-1);
      }
  }

  __global__ void matrixPolyderNewLayout(const float *coefImg, float *coefImgDer, const int w, const int h, const int m, size_t yOffset){
    size_t x = threadIdx.x + blockDim.x * blockIdx.x;
    size_t y = threadIdx.y + blockDim.y * blockIdx.y;
    if(x >= w || y >= h) return;

    size_t xOffsetDer = m-1;
    size_t yOffsetDer = w*xOffsetDer;

    size_t xOffsetCoef = m;
    size_t yOffsetCoef = w*xOffsetCoef;

    for (int i = 0; i < m - 1; ++i) //if of degree d=2, we have n=3 coeffs ax'2 + bx +c
      {
	size_t idxDer = x*xOffsetDer + y*yOffsetDer + i;
	size_t idxCoef = x*xOffsetCoef + y*yOffsetCoef + i;      

	coefImgDer[idxDer]=coefImg[idxCoef]*(m-i-1);
      }
  }

  //same as matlabs polyval(coefVec, x0) with n=length(coefVec), possibly strided over layer_size elements
  __host__ __device__ float polyval(const float *coefVec, const float x0, const int nc, const int layer_size) {
    float y0 = 0;
    for (int i = 0; i < nc - 1; ++i) //if of degree d=2, we have n=3 coeffs ax'2 + bx +c
      {
	y0 = (y0 + coefVec[i*layer_size])*x0; //Horner Shema 
      }
    y0 += coefVec[(nc-1)*layer_size]; //the last coefficient ist not multiplied by x, just by 1

    return y0;
  }

  __global__ void matrixPolyval(const float *coefImg, const float *x0Img, float *y0Img, const int w, const int h, const int nc) { //nc = d+1
    size_t x = threadIdx.x + blockDim.x * blockIdx.x;
    size_t y = threadIdx.y + blockDim.y * blockIdx.y;
    if(x >= w || y >= h) return;

    float x0=x0Img[id(x,y,w,h,0)];

    //1. option fixed size:
    //    const int Nmax = 14; //nc must be smaller than Nmax
    //    float coefVec[Nmax];
    //    for (int i = 0; i < nc; ++i)
    //    {
    //    	coefVec[i]=coefImg[id(x,y,w,h,i)];
    //    }
    // 	y0Img[id(x,y,w,h,0)] = polyval(coefVec,x0,nc);

    //2. option: copy and adapt code from polyval
    //    	float y0 = 0;
    // 		for (int i = 0; i < nc - 1; ++i) //if of degree d=2, we have n=3 coeffs ax'2 + bx +c
    // 		{
    // 			y0 = (y0 + coefImg[id(x,y,w,h,i)])*x0; //Horner Schema 
    // 		}
    // 		y0 += coefImg[id(x,y,w,h,nc - 1)]; //the last coefficient ist not multiplied by x, just by 1
    //    	y0Img[id(x,y,w,h,0)] = y0;
   

    //3. option: device function with pointer and strided access
    float* coefVec = (float*) coefImg + id(x,y,w,h,0);
    y0Img[id(x,y,w,h,0)] = polyval(coefVec,x0,nc,w*h);
  }

  __global__ void matrixPolyvalNewLayout(const float *coefImg, const float *x0Img, float *y0Img, const int w, const int h, const int nc) { //nc = d+1
    size_t x = threadIdx.x + blockDim.x * blockIdx.x;
    size_t y = threadIdx.y + blockDim.y * blockIdx.y;
    if(x >= w || y >= h) return;

    float x0=x0Img[id(x,y,w,h,0)];

    size_t xOffset = nc;
    size_t yOffset = w*xOffset;

    const float* coefVec = &coefImg[x*xOffset + y*yOffset];
    y0Img[id(x,y,w,h,0)] = polyval(coefVec,x0,nc,1);
  }

  //equivalent to matlab's [maxVal, depthImg] = max(sharpness,[],3); 
  __global__ void findMax(const float *values, float *maxValues, float *indicesMaxValues, const int w, const int h, const int nc) {
    size_t x = threadIdx.x + blockDim.x * blockIdx.x;
    size_t y = threadIdx.y + blockDim.y * blockIdx.y;
    //size_t z = threadIdx.z + blockDim.z * blockIdx.z;
    if(x >= w || y >= h /*|| z >= nc*/) return;

    float maxValue=values[id(x,y,w,h,0)];
    int maxIdx=0;

    //really simple solution which just iterates as a start
    for (int i = 1; i < nc; ++i)
      {
	float v=values[id(x,y,w,h,i)];
	if(v>maxValue){
	  maxValue=v;
	  maxIdx=i;
	}
      }

    maxValues[id(x,y,w,h,0)]=maxValue;

#ifdef CUDA_MATLAB
    maxIdx++; //matlab uses 1 based indexing
#endif

    indicesMaxValues[id(x,y,w,h,0)]=maxIdx; //int value is saved in float array to make things easier. Like this. we just have one datatype everywhere


  }

  __global__ void findMaxIndices(const float *values, float *indicesMaxValues, const int wValues, const int hValues, const int nc,
				 const size_t wIndices, const size_t hIndices, const size_t yOffset) {
    size_t x = threadIdx.x + blockDim.x * blockIdx.x;
    size_t y = threadIdx.y + blockDim.y * blockIdx.y;
    //size_t z = threadIdx.z + blockDim.z * blockIdx.z;
    if(x >= wValues || y >= hValues /*|| z >= nc*/) return;

    size_t index = x + y*wIndices + yOffset*wIndices;
    if (index >= wIndices*hIndices)
      return;  

    float maxValue=values[id(x,y,wValues,hValues,0)];
    int maxIdx=0;

    //really simple solution which just iterates as a start
    for (int i = 1; i < nc; ++i)
      {
	float v=values[id(x,y,wValues,hValues,i)];
	if(v>maxValue){
	  maxValue=v;
	  maxIdx=i;
	}
      }

#ifdef CUDA_MATLAB
    maxIdx++; //matlab uses 1 based indexing
#endif
    indicesMaxValues[index] = static_cast<float>(maxIdx);
  }

  /*
    equivalent to following matlab code:
    % possibly reduce the effect of strong edges deciding everything:
    denomRegu = 0.1;
    [maxVal, depthImg] = max(sharpness,[],3);  //get maxVal and depthImg using kernel findMax from above
    for i=1:nrImgs
    sharpness(:,:,i) = sharpness(:,:,i)./(maxVal + denomRegu);
    end
  */
  __global__ void scaleSharpnessValuesGPU(const float *maxValues, float *sharpness, const int w, const int h, const int nc, const float denomRegu) {
    size_t x = threadIdx.x + blockDim.x * blockIdx.x;
    size_t y = threadIdx.y + blockDim.y * blockIdx.y;
    if(x >= w || y >= h) return;

    float scaling = 1.0f / ( maxValues[id(x,y,w,h,0)] + denomRegu );

    for (int i = 0; i < nc; ++i)
      {
	sharpness[id(x,y,w,h,i)] *= scaling; // * is a little faster than / (1.323590 seconds vs. 1.357984 seconds for [1080 x 1920 x 20] float image (around 150MB))
      }
  }

  __global__ void cuFCTinitIndex(size_t *v_index, int w, int h){
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;

    if (i >= w || j >= h)
      return;

    int w2 = (w-1) / 2;
    int h2 = (h-1) / 2;
    size_t tmp;
    if (i <= w2 && j <= h2) {
      tmp=id2(2*i,2*j,w,h);
    } else if (i > w2 && j <= h2) {
      tmp=id2(2*w-2*i-1,2*j,w,h);
    } else if (i <= w2 && j > h2) {
      tmp=id2(2*i,2*h-2*j-1,w,h);
    } else if (i > w2 && j > h2) {
      tmp=id2(2*w-2*i-1,2*h-2*j-1,w,h);
    }
    v_index[id2(i,j,w,h)]=tmp;
  }

  __global__ void cuFCTinitRoots(cuFloatComplex *roots, int w, int h){
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    //these constants could potentially be in the constant memory, but they would already fill the whole constant memory for hd images, so we decided against it
    if (i >= 2*(w+h))
      return;

    //precompute e^(2*PI*I*i/(4*N)) = cos(2*PI*I*i/(4*N)) - I*sin(2*PI*I*i/(4*N)) for N=w and N=h
    float re,im;
    if(i<2*w) {
      re=cos(M_PI*(i-w)/(2*w));
      if(abs(re)<0.000001) re=0;
      im=-sin(M_PI*(i-w)/(2*w));
      if(abs(im)<0.000001) im=0;
    } else {
      re=cos(M_PI*(i-w-w-h)/(2*h));
      if(abs(re)<0.000001) re=0;
      im=-sin(M_PI*(i-w-w-h)/(2*h));
      if(abs(im)<0.000001) im=0;
    }
    roots[i]=make_cuComplex(re,im);
  }

  //step 1 FCT - resort input
  __global__ void resort2DArrayForward(float *x, float *v, size_t *v_index, int w, int h) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;

    if (i >= w || j >= h)
      return;

    int index = id2(i, j, w, h);
    v[index] = x[v_index[index]];
  }

  //step 3 FCT - throw away unneeded FFT stuff, scale and fill output
  __global__ void cuFCTCalcOutput(cuFloatComplex *v, float *c, cuFloatComplex *roots, int w, int h) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;

    if (i >= w || j >= h)
      return;

    int w2=w/2+1;
  
    //Black magic starts here.
    cuFloatComplex sum1, sum2;
    if(i<w2){
      int tmp=id2(i,j,w2,h);
      sum1 = cuCmulf(v[tmp], roots[w + w + h + j]);
      //for j=0: (h-j)=h and the point (i,h) is not in our image. instead replicate point i,0, which results in the same as sum1
      tmp=id2(i, h-j, w2, h);
      sum2 = (j == 0) ? sum1 : cuCmulf(v[tmp], roots[w + w + h - j]);
    } else {
      //use hermitian symmetry of the fft output (2nd half of fft does not need to be calculated)
      int tmp= (j == 0) ? id2(w - i, 0, w2, h) : id2(w - i, h - j, w2, h);
      sum1 = cuCmulf(cuConjf(v[tmp]), roots[w + w + h + j]);
      tmp=id2(w-i, j, w2, h);
      sum2 = cuCmulf(cuConjf(v[tmp]), roots[w + w + h - j]);
    }
    //Black magic ends here.

    //the scaling is not neccessary/part of the specification and breaks the direct correspondence with fft
    //but this scaling makes the matrix orthogonal and is the same as matlab/JPEG use
    float alpha=1.0/sqrtf(w*h);
    if(i==0) alpha/=sqrtf(2.0);
    if(j==0) alpha/=sqrtf(2.0);

    c[id2(i,j,w,h)]=alpha*cuCrealf(cuCmulf(roots[w + i], cuCaddf(sum1,sum2)));
  }

  //step 1 IFCT - undo scaling, reorder into v
  __global__ void cuIFCTPrepareInput(float *c, cuFloatComplex *v, cuFloatComplex *roots, int w, int h) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;

    int w2=w/2+1;

    if (i >= w2 || j >= h)
      return;

    //undo alpha from step 3 fct to get a direct correspondence to fft back
    float alpha=0.5/sqrtf(w*h);

    //Black magic starts here
    float re,im=0;
    re=c[id2(i,j,w,h)];
    if(j==0) {
      if(i==0) alpha*=2;
      else {
        alpha*=sqrtf(2.0);
        im=-c[id2(w-i,j,w,h)];
      }
    } else {
      im=-c[id2(i,h-j,w,h)];
      if(i==0) {
        alpha*=sqrtf(2.0);
      } else {
        re-=c[id2(w-i,h-j,w,h)];
        im-=c[id2(w-i,j,w,h)];
      }
    }
    //Black magic ends here.

    v[id2(i,j,w2,h)] = cuCmulf(cuCmulf(make_cuFloatComplex(re*alpha, im*alpha), roots[w+w+h-j]), roots[w-i]);
  }

  //step 3 IFCT - resort and use only the relevant data
  __global__ void resort2DArrayBackward(float *v, float *x, size_t *v_index, int w, int h) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;

    if (i >= w || j >= h)
      return;

    int index = id2(i, j, w, h);

    x[v_index[index]]=v[index];
  }
}