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

#include <LinearizedADMM.cuh>
#include <openCVHelpers.h>

#include <CPUTimer.h>
#include <cudaWrappers.h>

#include <vector>
#include <iostream>
#include <cstdio>

using namespace std;
using namespace cv;

namespace vdff {
  LinearizedADMM::LinearizedADMM(size_t w, size_t h, float min, float max) 
    : nrBytes(w * h * sizeof(float)),minVal(min), maxVal(max) {
    // init LaplaceInversion class
    vector<int> size;
    size.push_back(h);
    size.push_back(w);

    vector<int> hvec(2, 1);
    eyePlusLaplace = new LaplaceInversion(size, hvec);

    info.w = w;
    info.h = h;
  }

  void LinearizedADMM::allocateCUDAMemory() {
    cudaMalloc((void**)&d_u, nrBytes); CUDA_CHECK;
    cudaMemcpy(d_u, l_u, nrBytes, cudaMemcpyHostToDevice); CUDA_CHECK;

    // reset variables to zero
    cudaMalloc((void**)&d_ux, nrBytes); CUDA_CHECK;
    cudaMemset(d_ux, 0, nrBytes); CUDA_CHECK;
  
    cudaMalloc((void**)&d_uy, nrBytes); CUDA_CHECK;
    cudaMemset(d_uy, 0, nrBytes); CUDA_CHECK;
  
    cudaMalloc((void**)&d_dx, nrBytes); CUDA_CHECK;
    cudaMemset(d_dx, 0, nrBytes); CUDA_CHECK;
  
    cudaMalloc((void**)&d_dy, nrBytes); CUDA_CHECK;
    cudaMemset(d_dy, 0, nrBytes); CUDA_CHECK;

    cudaMalloc((void**)&d_bx, nrBytes); CUDA_CHECK;
    cudaMemset(d_bx, 0, nrBytes); CUDA_CHECK;

    cudaMalloc((void**)&d_by, nrBytes); CUDA_CHECK;
    cudaMemset(d_by, 0, nrBytes); CUDA_CHECK;

    // ux + bx and uy + by
    cudaMalloc((void**)&d_sum_ux_bx, nrBytes); CUDA_CHECK;
    cudaMemset(d_sum_ux_bx, 0, nrBytes); CUDA_CHECK;

    cudaMalloc((void**)&d_sum_uy_by, nrBytes); CUDA_CHECK;
    cudaMemset(d_sum_uy_by, 0, nrBytes); CUDA_CHECK;

    // ux - dx and uy - dy
    cudaMalloc((void**)&d_subtract_ux_dx, nrBytes); CUDA_CHECK;
    cudaMemset(d_subtract_ux_dx, 0, nrBytes); CUDA_CHECK;

    cudaMalloc((void**)&d_subtract_uy_dy, nrBytes); CUDA_CHECK;
    cudaMemset(d_subtract_uy_dy, 0, nrBytes); CUDA_CHECK;

    // dx - bx and dy - by
    cudaMalloc((void**)&d_subtract_dx_bx, nrBytes); CUDA_CHECK;
    cudaMemset(d_subtract_dx_bx, 0, nrBytes); CUDA_CHECK;

    cudaMalloc((void**)&d_subtract_dy_by, nrBytes); CUDA_CHECK;
    cudaMemset(d_subtract_dy_by, 0, nrBytes); CUDA_CHECK;

    // space for x- and y-Derivative for (dx - bx) and (dy - by)
    cudaMalloc((void**)&d_dxDxBx, nrBytes); CUDA_CHECK;
    cudaMemset(d_dxDxBx, 0, nrBytes); CUDA_CHECK;

    cudaMalloc((void**)&d_dyDyBy, nrBytes); CUDA_CHECK;
    cudaMemset(d_dyDyBy, 0, nrBytes); CUDA_CHECK;        
  }

  LinearizedADMM::~LinearizedADMM() {
    delete eyePlusLaplace;
    freeHostMemory();
  }

  void LinearizedADMM::freeCUDAMemory() {
    cudaFree(d_u);
    
    cudaFree(d_ux);
    cudaFree(d_uy);
  
    cudaFree(d_dx);
    cudaFree(d_dy);
  
    cudaFree(d_bx);
    cudaFree(d_by);

    cudaFree(d_sum_ux_bx);
    cudaFree(d_sum_uy_by);

    cudaFree(d_subtract_ux_dx);
    cudaFree(d_subtract_uy_dy);

    cudaFree(d_subtract_dx_bx);
    cudaFree(d_subtract_dy_by);
  
    cudaFree(d_dxDxBx);
    cudaFree(d_dyDyBy);        
  }

  void LinearizedADMM::freeHostMemory() {
    delete[] l_u;
  }

  void LinearizedADMM::updateU(dim3 gridSize, dim3 blockSize, float *d_coefsDerivative, size_t degreeDerivatives, float lambda, float tau, float dataFidelityParam) {
    cudaSubtractArrays(gridSize, blockSize, d_dx, d_bx, d_subtract_dx_bx, info.w, info.h, info.nc);
    cudaSubtractArrays(gridSize, blockSize, d_dy, d_by, d_subtract_dy_by, info.w, info.h, info.nc);
    cudaDeviceSynchronize(); CUDA_CHECK;
    
    // CAUTION: we have to later multiply it with -lambda, since in the matlab code [1 -1 0] is
    // used instead of the 'normal' backward differences [-1 1 0]
    cudaCalcBackwardDifferencesXDirection(gridSize, blockSize, d_subtract_dx_bx, d_dxDxBx, info.w, info.h, info.nc, ZERO); CUDA_CHECK;
    cudaCalcBackwardDifferencesYDirection(gridSize, blockSize, d_subtract_dy_by, d_dyDyBy, info.w, info.h, info.nc, ZERO); CUDA_CHECK;
    cudaDeviceSynchronize(); CUDA_CHECK;

    // Compute gradient descent step -- NEW!
    //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    float *d_evaluatedDerivatives, *d_gradientDescentStep;
    float *d_solveTerm, *d_addDxBxDyBy, *d_updateStep;
  
    cudaMalloc(&d_evaluatedDerivatives, nrBytes); CUDA_CHECK;
    cudaMalloc(&d_gradientDescentStep, nrBytes); CUDA_CHECK;
    cudaMalloc(&d_solveTerm, nrBytes); CUDA_CHECK;
    cudaMalloc(&d_addDxBxDyBy, nrBytes); CUDA_CHECK;
    cudaMalloc(&d_updateStep, nrBytes); CUDA_CHECK;

    cudaPolyval(gridSize, blockSize, d_coefsDerivative, d_u, d_evaluatedDerivatives, info.w, info.h, degreeDerivatives);
    cudaDeviceSynchronize(); CUDA_CHECK;
  
    cudaMultiplyArrayWithScalar(gridSize, blockSize, d_evaluatedDerivatives, -tau * dataFidelityParam, d_gradientDescentStep, info.w, info.h, info.nc);
    cudaDeviceSynchronize(); CUDA_CHECK;

    cudaMultiplyArrayWithScalar(gridSize, blockSize, d_dxDxBx, -lambda, d_dxDxBx, info.w, info.h, info.nc);
    cudaMultiplyArrayWithScalar(gridSize, blockSize, d_dyDyBy, -lambda, d_dyDyBy, info.w, info.h, info.nc);
    cudaDeviceSynchronize(); CUDA_CHECK;

    cudaAddArrays(gridSize, blockSize, d_dxDxBx, d_dyDyBy, d_addDxBxDyBy, info.w, info.h, info.nc);
    cudaDeviceSynchronize(); CUDA_CHECK;

    cudaAddArrays(gridSize, blockSize, d_addDxBxDyBy, d_gradientDescentStep, d_updateStep, info.w, info.h, info.nc);
    cudaDeviceSynchronize(); CUDA_CHECK;

    cudaAddArrays(gridSize, blockSize, d_u, d_updateStep, d_solveTerm, info.w, info.h, info.nc);
    cudaDeviceSynchronize(); CUDA_CHECK;

    eyePlusLaplace->solveGPU(d_solveTerm, d_u, info.w, info.h, info.nc);

    cudaFree(d_solveTerm);
    cudaFree(d_evaluatedDerivatives);  
    cudaFree(d_addDxBxDyBy);
    cudaFree(d_gradientDescentStep);
    cudaFree(d_updateStep);
  }

  void LinearizedADMM::thresholdU(dim3 gridSize, dim3 blockSize, float thresholdLow, float thresholdHigh) {
    thresholdArray<<<gridSize, blockSize>>>(d_u, thresholdLow, thresholdHigh, d_u, info.w, info.h, info.nc); CUDA_CHECK;  
  }

  void LinearizedADMM::calcUxUy(dim3 gridSize, dim3 blockSize, BoundaryBehavior behavior) {
    cudaCalcForwardDifferences(gridSize, blockSize, d_u, d_ux, d_uy, info.w, info.h, info.nc, behavior); CUDA_CHECK;  
  }

  void LinearizedADMM::updateG(dim3 gridSize, dim3 blockSize, float alpha) {
    // first calculate the sum ux+bx and uy+by; this can be done in parallel, no need to synchronize
    cudaAddArrays(gridSize, blockSize, d_ux, d_bx, d_sum_ux_bx, info.w, info.h, info.nc); CUDA_CHECK;
    cudaAddArrays(gridSize, blockSize, d_uy, d_by, d_sum_uy_by, info.w, info.h, info.nc); CUDA_CHECK;
    // now we need the synchronize
    cudaDeviceSynchronize(); CUDA_CHECK;
    
    // calculate update step for g (in paper) - closed form solution is isoShrinkage
    isoShrinkage<<<gridSize, blockSize>>>(d_sum_ux_bx, d_sum_uy_by, d_dx, d_dy, alpha, info.w, info.h, info.nc); CUDA_CHECK;  
  }

  void LinearizedADMM::updateB(dim3 gridSize, dim3 blockSize) {
    // compute subtraction
    cudaSubtractArrays(gridSize, blockSize, d_ux, d_dx, d_subtract_ux_dx, info.w, info.h, info.nc);
    cudaSubtractArrays(gridSize, blockSize, d_uy, d_dy, d_subtract_uy_dy, info.w, info.h, info.nc);
    cudaDeviceSynchronize(); CUDA_CHECK; //TODO(Dennis): could be further improved

    // compute update step for b
    cudaAddArrays(gridSize, blockSize, d_bx, d_subtract_ux_dx, d_bx, info.w, info.h, info.nc); CUDA_CHECK;
    cudaAddArrays(gridSize, blockSize, d_by, d_subtract_uy_dy, d_by, info.w, info.h, info.nc); CUDA_CHECK;  
  }

  void LinearizedADMM::scaleBWithFactor(dim3 gridSize, dim3 blockSize, float scalar) {
    // bx = bx / fac 
    cudaMultiplyArrayWithScalar(gridSize, blockSize, d_bx, scalar, d_bx, info.w, info.h, info.nc); CUDA_CHECK;
    // by = by / fac
    cudaMultiplyArrayWithScalar(gridSize, blockSize, d_by, scalar, d_by, info.w, info.h, info.nc); CUDA_CHECK;  
  }

  cv::Mat LinearizedADMM::run(float *d_energyDerivative, size_t derivativeDegree, float dataFidelityParam, float tau,
			      const Mat& init, bool plotIterations, size_t convIter, size_t maxIter, float lambda) {
    assert(init.cols == info.w && init.rows == info.h);
    assert(init.channels() == 1);
    info.nc = init.channels();

    vector<float> timeUpdateU;
    vector<float> timeThresholdU;
    vector<float> timeCalcUxUy;
    vector<float> timeUpdateG;
    vector<float> timeUpdateB;
    vector<float> timeOneIteration;

    CPUTimer t;
    CPUTimer iterationTimer;

    // set lambda of laplace inversion class
    eyePlusLaplace->setLambda(lambda);
  
    // --- CUDA stuff (alloc and memcpy)
    dim3 blockSize(32, 8, 1);
    dim3 gridSize((info.w + blockSize.x - 1) / blockSize.x, (info.h + blockSize.y - 1) / blockSize.y, 1);

    l_u = new float[info.w * info.h * info.nc];
    openCVHelpers::convert_mat_to_layered(l_u, init);

    allocateCUDAMemory();
  
    // dbg - just do it once
    for (size_t it = 0; it < maxIter; ++it) {
      iterationTimer.tic();

      // copy d_u back to mat
      t.tic();
      // u is of type cv::Mat
      updateU(gridSize, blockSize, d_energyDerivative, derivativeDegree, lambda, tau, dataFidelityParam);
      timeUpdateU.push_back(t.tocInSeconds());
    
      // threshold u so that u in [minVal, maxVal]
      t.tic();
      thresholdU(gridSize, blockSize, minVal, maxVal);
      cudaDeviceSynchronize(); CUDA_CHECK;
      timeThresholdU.push_back(t.tocInSeconds());

      // calculate forward differences on updated u (aka u^(k+1))
      t.tic();
      calcUxUy(gridSize, blockSize, REPLICATE);
      cudaDeviceSynchronize(); CUDA_CHECK;
      timeCalcUxUy.push_back(t.tocInSeconds());

      // update g (has closed form in isoShrinkage)
      t.tic();
      updateG(gridSize, blockSize, tau/lambda);
      cudaDeviceSynchronize(); CUDA_CHECK;
      timeUpdateG.push_back(t.tocInSeconds());

      // update scaled dual variable b
      t.tic();
      updateB(gridSize, blockSize);
      cudaDeviceSynchronize(); CUDA_CHECK;
      timeUpdateB.push_back(t.tocInSeconds());

      // do we visualize? - then copy stuff back
      if (plotIterations /*&& it % 50 == 0*/) {
	cudaMemcpy(l_u, d_u, nrBytes, cudaMemcpyDeviceToHost); CUDA_CHECK;

	Mat vis = Mat::zeros(init.rows, init.cols, CV_32FC1);
	openCVHelpers::convert_layered_to_mat(vis, l_u);
      
	//char buf[256];
	//sprintf(buf, "Iteration %zd", it);
	openCVHelpers::showDepthImage("Current Iteration", vis, 100 + 2*(init.cols + 80), 500);      
      
	char key = Utils::waitKey2(1,false);
	int keyNr = static_cast<int>(key);
	if(keyNr == 27 || keyNr == 'q' || key == 'Q') {
	  cout << "Aborting at iteration " << it << endl;
	  break;
	}
      }
      cout << "\r" << flush;
      cout << "Iteration " << it << " of " << maxIter;      
    

      // implements section C.) of the paper
      if (it > convIter) {
	float fac = 1.02f;
	lambda *= fac;

	// b = b * 1.0f/fac
	scaleBWithFactor(gridSize, blockSize, 1.0f/fac);

	// update (lambdaK^TK + I)
	eyePlusLaplace->setLambda(lambda);
      
	cudaDeviceSynchronize(); CUDA_CHECK;
      }
      timeOneIteration.push_back(iterationTimer.tocInSeconds());
    }
    // do new line after the last iteration 
    cout << endl;

    Mat res = Mat::zeros(init.rows, init.cols, init.type());
    cudaMemcpy(l_u, d_u, nrBytes, cudaMemcpyDeviceToHost); CUDA_CHECK;
    openCVHelpers::convert_layered_to_mat(res, l_u);
  
    // --- free CUDA memory
    freeCUDAMemory();

    // print stats for linearized admm
    cout << "Average time for updating U: " << Utils::getAverage(timeUpdateU) << " s" << endl;
    cout << "Average time for thresholding U: " << Utils::getAverage(timeThresholdU) << " s" << endl;
    cout << "Average time for calculating Ux and Uy: " << Utils::getAverage(timeCalcUxUy) << " s" << endl;
    cout << "Average time for updating G: " << Utils::getAverage(timeUpdateG) << " s" << endl;
    cout << "Average time for updating B: " << Utils::getAverage(timeUpdateB) << " s" << endl;
    cout << "Average time for one iteration of ADMM: " << Utils::getAverage(timeOneIteration) << " s" << endl;

    return res;  
  }
}