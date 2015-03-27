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

#ifndef LINEARIZED_ADMM_H
#define LINEARIZED_ADMM_H

#include <LaplaceInversion.cuh>
#include <opencv2/core/core.hpp>

#include <DataPreparator.cuh>
#include <common_kernels.cuh>
#include <utils.cuh>

namespace vdff {
  class LinearizedADMM {
  private:

    float *l_u;
    float *d_u;
    float *d_ux, *d_uy;
    float *d_dx, *d_dy;
    float *d_bx, *d_by;

    float *d_sum_ux_bx, *d_sum_uy_by;
    float *d_subtract_ux_dx, *d_subtract_uy_dy;
    float *d_subtract_dx_bx, *d_subtract_dy_by;

    float minVal, maxVal;
    float scale;
    size_t nrBytes;
    Utils::ImgInfo info;

    // x-derivative of (dx - bx) and y-derivative of (dy - by)
    float *d_dxDxBx, *d_dyDyBy;  
    LaplaceInversion *eyePlusLaplace;

    void initLaplaceInversionClass(size_t w, size_t h, float lambda);
  
    void allocateCUDAMemory();
    void freeCUDAMemory();
    void freeHostMemory();

    // update steps for every iteration
    void updateU(dim3 gridSize, dim3 blockSize, float *d_coefsDerivative, size_t degreeDerivatives, float lambda, float tau, float dataFidelityParam);
    void thresholdU(dim3 gridSize, dim3 blockSize, float thresholdLow, float thresholdHigh);
    void calcUxUy(dim3 gridSize, dim3 blockSize, BoundaryBehavior behavior=REPLICATE);
    void updateG(dim3 gridSize, dim3 blockSize, float alpha);
    void updateB(dim3 gridSize, dim3 blockSize);
    void scaleBWithFactor(dim3 gridSize, dim3 blockSize, float scalar);
  
  public:
    LinearizedADMM(size_t w, size_t h, float min, float max);
    ~LinearizedADMM();

    cv::Mat run(float *d_energyDerivative, size_t derivativeDegree, float dataFidelityParam, float tau,
		const cv::Mat& init, bool plotIterations, size_t convIter=350, size_t maxIter=400, float lambda=50.0f);
  };
}
#endif
