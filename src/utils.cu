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

#include <utils.cuh>
#include <cstring>
#include <stdio.h>

#ifndef __USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#include <math.h>
#endif
#include <cstdlib>

using namespace std;

namespace vdff {
  string prev_file = "";
  int prev_line = 0;
  void cuda_check(string file, int line)
  {
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
      {
	cout << endl << file << ", line " << line << ": " << cudaGetErrorString(e) << " (" << e << ")" << endl;
	if (prev_line>0) cout << "Previous CUDA call:" << endl << prev_file << ", line " << prev_line << endl;
	exit(1);
      }
    prev_file = file;
    prev_line = line;
  }

  namespace Utils {

    void printTiming(CUDATimer &timer, const string& launchedKernel) {
      cout << "Elapsed time";
  
      if (!launchedKernel.empty())
	cout << " for " << launchedKernel;

      cout << ": " << timer.toc() << " ms" << endl;
    }

    float getAverage(const vector<float> &v) {
      float sum = 0.0f;
      for (size_t i = 0; i < v.size(); ++i)
	sum += v[i];
  
      return sum / v.size();
    }

    cudaDeviceProp queryDeviceProperties() {
      int nrDevices;
      cudaGetDeviceCount(&nrDevices); CUDA_CHECK;

      cudaDeviceProp bestProp;
      // check for largest constant memory
      size_t maxConstantMemory = 0;

      for(int i = 0; i < nrDevices; ++i) {
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, i);

	if (prop.totalConstMem > maxConstantMemory) {
	  maxConstantMemory = prop.totalConstMem;
	  bestProp = prop;
	}
      }

      return bestProp;
    }

    void getAvailableGlobalMemory(size_t *free, size_t *total, bool print) {
      cudaMemGetInfo(free, total); CUDA_CHECK;
      if(print){
        printf("AvailableGlobalMemory: %0.5f / %0.5f MB\n",*free/1e6f,*total/1e6f);
      }
    }

    void memprint() {
      size_t free,total;
      cudaMemGetInfo(&free,&total); CUDA_CHECK;
      printf("AvailableGlobalMemory: %0.5f / %0.5f MB\n",free/1e6f,total/1e6f);
    }
  }
}
