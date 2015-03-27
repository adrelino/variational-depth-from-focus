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

#ifndef UTILS_H
#define UTILS_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <iostream>
#include <vector>
#include <CUDATimer.h>
#include <stdio.h>

namespace vdff {
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

  // cuda error checking
#define CUDA_CHECK cuda_check(__FILE__,__LINE__)
  void cuda_check(std::string file, int line);

  namespace Utils {

    typedef struct {
      int w;
      int h;
      int nc;
    } ImgInfo;

    typedef struct {
      int w;
      int h;
      int nc;
      int nrImgs;
      void print(){
	printf("InfoImqSeq:   [%d x %d x (%d * %d)] [w x h x (nc * nrImgs)]\n",w,h,nc,nrImgs);
      };
    } InfoImgSeq;

    cudaDeviceProp queryDeviceProperties();
    void printTiming(CUDATimer &timer, const std::string& launchedKernel="");

    float getAverage(const std::vector<float> &v);
    void getAvailableGlobalMemory(size_t *free, size_t *total, bool print=false);
    void memprint();
  }
}
#endif

  
