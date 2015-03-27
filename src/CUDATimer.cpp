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

#include <CUDATimer.h>
#include <utils.cuh>

#include <iostream>
using namespace std;

namespace vdff {
  CUDATimer::CUDATimer() : startWasSet(false), stopWasSet(true) {
    cudaEventCreate(&startTime); CUDA_CHECK;
    cudaEventCreate(&stopTime); CUDA_CHECK;
  }

  CUDATimer::~CUDATimer() {
    cudaEventDestroy(startTime); CUDA_CHECK;
    cudaEventDestroy(stopTime); CUDA_CHECK;
  }

  void CUDATimer::tic(cudaStream_t stream) {
    assert(stopWasSet && !startWasSet);
    this->stream = stream;
  
    cudaEventRecord(startTime, this->stream); CUDA_CHECK;

    startWasSet = true;
    stopWasSet = false;
  }

  float CUDATimer::toc() {
    assert(startWasSet && !stopWasSet);

    // just in case if assertions are disabled
    if (!startWasSet)
      return 0.0f;
  
    cudaEventRecord(stopTime, this->stream); CUDA_CHECK;
    // block CPU until the stop event was recorded
    cudaEventSynchronize(stopTime); CUDA_CHECK;

    // get elapsed time in ms
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, startTime, stopTime); CUDA_CHECK;

    stopWasSet = true;
    startWasSet = false;

    return elapsedTime;
  }
}
