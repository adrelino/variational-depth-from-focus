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

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
#include <functional>
#include <numeric>

#include <LaplaceInversion.cuh>
#include <FCT.cuh>
#include <openCVHelpers.h>
#include <common_kernels.cuh>
#include <utils.cuh>

using namespace cv;
using namespace std;

LaplaceInversion::LaplaceInversion() : transform(256, 256), d_denom(NULL){
  size.push_back(256);
  size.push_back(256);

  hvec.push_back(1);
  hvec.push_back(1);
  
  initKernelGPU();
  cudaDeviceSynchronize(); CUDA_CHECK;
  setLambda(1.0f);
  cudaDeviceSynchronize(); CUDA_CHECK;    
}

LaplaceInversion::LaplaceInversion(const int *sz, int sizeDim,
				   const int *hv, int hvecDim) : transform(sz[1], sz[0]),
								 size(sz, sz + sizeDim), hvec(hv, hv + hvecDim),
								 d_denom(NULL) {
  assert(size.size() == hvec.size());
  this->lambda = 1.0f;
  
  initKernelGPU();
  cudaDeviceSynchronize(); CUDA_CHECK;
  setLambda(1.0f);
}

LaplaceInversion::LaplaceInversion(const vector<int> size, const vector<int> hvec) : transform(size[1], size[0]),
										     d_denom(NULL) {
  assert(size.size() == hvec.size());

  this->size = size;
  this->hvec = hvec;

  initKernelGPU();
  cudaDeviceSynchronize(); CUDA_CHECK;
  setLambda(1.0f);
}

LaplaceInversion::~LaplaceInversion() {
  cudaFree(d_kernel);
  cudaFree(d_denom);
}

void LaplaceInversion::setLambda(float lambda) {
  this->lambda = lambda;

  // update our denominator used for solve and solveGPU
  denom = (1 + lambda * mKernel);

  int w = size[1];
  int h = size[0];

  dim3 blockSize(32, 8, 1);
  dim3 gridSize((w + blockSize.x -1) / blockSize.x, (h + blockSize.y - 1) / blockSize.y, 1);

  if (d_denom == NULL)
    cudaMalloc(&d_denom, w * h * sizeof(float)); CUDA_CHECK;
  
  multiplyArrayWithScalar<<<gridSize, blockSize>>>(d_kernel, lambda, d_denom, w, h, 1); CUDA_CHECK;
  cudaDeviceSynchronize(); CUDA_CHECK;
  addScalarToArray<<<gridSize, blockSize>>>(d_denom, 1.0f, d_denom, w, h, 1); CUDA_CHECK;
  cudaDeviceSynchronize(); CUDA_CHECK;    
}

float LaplaceInversion::getLambda() {
  return lambda;
}

Mat LaplaceInversion::getKernel() {
  return mKernel;
}

// this method computes the matrix for the discrete laplacian
// See 
// http://en.wikipedia.org/wiki/Kronecker_sum_of_discrete_Laplacians
// for a more clear overview
// void LaplaceInversion::initKernel() {
//   size_t nrDimensions = size.size();
  
//   size_t gridElements = 1;
//   for(size_t i = 0; i < nrDimensions; ++i) {
//     gridElements *= size[i];
//   }
  
//   vector<int> repeats = vector<int>(nrDimensions, 1);
//   // do cumprod with first element 1
//   for(size_t i = 1; i < nrDimensions; ++i) {
//     repeats[i] = size[i-1] * repeats[i-1]; 
//   }
  
//   // set up dct kernel for inversion
//   kernel = new float[gridElements];
//   memset(kernel, 0.0f, gridElements * sizeof(float));

//   for (size_t dim = 0; dim < nrDimensions; ++dim) {
//     int curSize = size[dim];    
//     float denom = static_cast<float>(hvec[dim]);
    
//     for(size_t k = 0; k < gridElements; ++k) {
//       int gridIdx = static_cast<int>(k / repeats[dim]);
//       gridIdx = gridIdx % curSize;
//       float val = sin((M_PI * gridIdx) / (2.0f * curSize)) / denom;
//       kernel[k] += 4*(val*val);       
//     }
//   }

//   // we are assuming that size is not bigger than 3D-dimensional
//   assert(nrDimensions <= 3);

//   // TODO(Dennis): currently just working for 2D-Kernels
//   // get later rid of this, no need for openCV Mat anymore
//   mKernel = reshapeColVector(kernel, size[0], size[1]);
// }

void LaplaceInversion::initKernelGPU() {
  size_t nrDimensions = size.size();

  int nrGridElements = 1;
  for(size_t i = 0; i < nrDimensions; ++i) {
    nrGridElements *= size[i];
  }
  
  vector<int> repeats = vector<int>(nrDimensions, 1);
  // do cumprod with first element 1
  for(size_t i = 1; i < nrDimensions; ++i) {
    repeats[i] = size[i-1] * repeats[i-1]; 
  }

  // set up dct kernel for inversion
  float *d_grid;
  int *d_size, *d_repeats, *d_hvec;

  size_t nrBytes = static_cast<size_t>((nrGridElements*nrDimensions) * sizeof(float));
  cudaMalloc(&d_grid, nrBytes); CUDA_CHECK;
  cudaMemset(d_grid, 0, nrBytes); CUDA_CHECK;

  cudaMalloc(&d_kernel, nrGridElements * sizeof(float)); CUDA_CHECK;
  cudaMemset(d_kernel, 0, nrGridElements * sizeof(float)); CUDA_CHECK;  

  cudaMalloc(&d_size, nrDimensions*sizeof(int)); CUDA_CHECK;
  cudaMemcpy(d_size, &size[0], nrDimensions*sizeof(int), cudaMemcpyHostToDevice); CUDA_CHECK;

  cudaMalloc(&d_repeats, nrDimensions*sizeof(int)); CUDA_CHECK;
  cudaMemcpy(d_repeats, &repeats[0], nrDimensions*sizeof(int), cudaMemcpyHostToDevice); CUDA_CHECK;

  cudaMalloc(&d_hvec, nrDimensions*sizeof(int)); CUDA_CHECK;
  cudaMemcpy(d_hvec, &hvec[0], nrDimensions*sizeof(int), cudaMemcpyHostToDevice); CUDA_CHECK;    

  dim3 blockSize(256, 1, 1);
  dim3 gridSize((nrGridElements + blockSize.x -1) / blockSize.x, nrDimensions, 1);
  createLaplacianKernelGrid<<<gridSize, blockSize>>>(d_grid, d_size, d_repeats, d_hvec, nrGridElements); CUDA_CHECK;
  cudaDeviceSynchronize(); CUDA_CHECK;

  dim3 gridSizeKernel((nrGridElements + blockSize.x -1) / blockSize.x, 1, 1);
  createLaplacianKernel<<<gridSizeKernel, blockSize>>>(d_grid, d_kernel, nrDimensions, nrGridElements); CUDA_CHECK;
  cudaDeviceSynchronize(); CUDA_CHECK;

  float *l_tmpKernel = new float[nrGridElements];
  float *l_correctKernel = new float[nrGridElements];

  cudaMemcpy(l_tmpKernel, d_kernel, nrGridElements*sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;
  // TODO: if we corrected the kernel, then we can just use convert_layered_to_mat
  mKernel = reshapeColVector(l_tmpKernel, size[0], size[1]);

  // transpose kernel
  size_t yCounter = 0;
  size_t xCounter = 0;

  for (size_t i = 0; i < nrGridElements; ++i) {
    l_correctKernel[i] = l_tmpKernel[xCounter + yCounter * size[0]];
    ++yCounter;
    
    if (yCounter % size[1] == 0) {
      yCounter = 0;
      ++xCounter;
    }
  }
  delete[] l_tmpKernel;
  
  // copy stuff back
  cudaMemcpy(d_kernel, l_correctKernel, nrGridElements*sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
  delete[] l_correctKernel;

  cudaFree(d_grid);
  cudaFree(d_size);
  cudaFree(d_repeats);
  cudaFree(d_hvec);  
}

Mat LaplaceInversion::solve(const Mat& f) {
  Mat res;

  // have f and kernel equal dimensions?
  if (f.channels() == 1 &&
      f.size() == mKernel.size()) {
    Mat fDCT;
    dct(f, fDCT);
    idct(fDCT / denom, res);
  }
  // TODO(Dennis): make this dependent of the actual dimension of the kernel,
  // but up to now we just have 2D-dimensional Laplacian kernels so now problem
  else if (f.channels() == 3){
    vector<Mat> channelsF = vector<Mat>(3);
    vector<Mat> channelsRes = vector<Mat>(3);    
    split(f, channelsF);

    for (int c = 0; c < f.channels(); ++c) {
      Mat fDCT;
      dct(channelsF[c], fDCT);
      idct(fDCT / denom, channelsRes[c]);
    }
    merge(channelsRes, res);
  } else {
    cerr << "Invalid dimension for laplace inversion!" << endl;      
  }
  return res;
}

void LaplaceInversion::solveGPU(float *d_f, float *d_res, size_t w, size_t h, size_t channels) {
  assert(w == size[1] && h == size[0]);

  size_t nrBytes = w * h * sizeof(float);
  float *d_fDCT;
  cudaMalloc(&d_fDCT, nrBytes); CUDA_CHECK;

  float *d_term;
  cudaMalloc(&d_term, nrBytes); CUDA_CHECK;

  dim3 blockSize(32, 8, 1);
  dim3 gridSize((w + blockSize.x -1) / blockSize.x, (h + blockSize.y - 1) / blockSize.y, 1);

  // have f and kernel equal dimensions?
  if (channels == 1) {
    transform.fct(d_f, d_fDCT); CUDA_CHECK;
    cudaDeviceSynchronize(); CUDA_CHECK;    
    divideArrays<<<gridSize, blockSize>>>(d_fDCT, d_denom, d_term, w, h, 1); CUDA_CHECK;
    cudaDeviceSynchronize(); CUDA_CHECK;    
    transform.ifct(d_term, d_res); CUDA_CHECK;
    cudaDeviceSynchronize(); CUDA_CHECK;
  }
  else if (channels == 3){
    for (size_t c = 0; c < channels; ++c) {
      transform.fct(&d_f[c*w*h], d_fDCT); CUDA_CHECK;
      cudaDeviceSynchronize(); CUDA_CHECK;      
      divideArrays<<<gridSize, blockSize>>>(d_fDCT, d_denom, d_term, w, h, 1); CUDA_CHECK;
      cudaDeviceSynchronize(); CUDA_CHECK;      
      transform.ifct(d_term, &d_res[c*w*h]); CUDA_CHECK;
      cudaDeviceSynchronize(); CUDA_CHECK;      
    }
  } else {
    cerr << "Invalid dimension for laplace inversion!" << endl;      
  }

  cudaFree(d_fDCT);
  cudaFree(d_term);
}


    

    
