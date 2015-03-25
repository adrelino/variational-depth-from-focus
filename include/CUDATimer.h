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

#ifndef CUDA_TIMER_H
#define CUDA_TIMER_H

#include <cuda.h>
#include <cuda_runtime.h>

class CUDATimer {
 private:
  cudaEvent_t startTime;
  cudaEvent_t stopTime;
  cudaStream_t stream;

  bool startWasSet;
  bool stopWasSet;
  
 public:
  CUDATimer();
  ~CUDATimer();

  // starts timer
  void tic(cudaStream_t streamID = 0);
  // stops timer and returns measured time in milliseconds
  float toc();
};

#endif
