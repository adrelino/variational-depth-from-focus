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

#include <LayeredMemory.cuh>
#include <opencv2/core/core.hpp>
#include <cuda.h>
#include <cudaWrappers.h>
#include <openCVHelpers.h>
#include <iostream>

using namespace std;
using namespace cv;

LayeredMemory::LayeredMemory(const char *dir, float minValue, float maxValue) : DataPreparator(dir, minValue, maxValue, LAYERED) {
}

LayeredMemory::~LayeredMemory() {
}

void LayeredMemory::copyMLAPIntoMemory(float *d_MLAPEstimate, size_t index) {
  cudaMemcpy(&l_sharpness[index*info.w*info.h*info.nc], d_MLAPEstimate, info.w*info.h*info.nc*sizeof(float),
	     cudaMemcpyDeviceToHost); CUDA_CHECK;
}

void LayeredMemory::copyMLAPIntoPageLockedMemory(float *d_MLAPEstimate, size_t index, const cudaStream_t streamID) {
  cudaMemcpyAsync(&l_sharpness[index*info.w*info.h*info.nc], d_MLAPEstimate, info.w*info.h*info.nc*sizeof(float),
		  cudaMemcpyDeviceToHost, streamID); CUDA_CHECK;
}

Mat LayeredMemory::findMaxSharpnessValues() {
  size_t nrPixels = info.w * info.h;

  l_maxValues = new float[nrPixels];
  l_indicesMaxValues = new float[nrPixels];

  // do it for every pixel
  // do find maximum and in the end scale it
  for(int y = 0; y < info.h; ++y) {
    cout << "\r" << flush;
    cout << "Find max sharpness of row " << (y+1) << " from " << info.h;    

    for (int x = 0; x < info.w; ++x) {
      float max = std::numeric_limits<float>::min();
      float maxIndex = -1.0f;

      for (int z = 0; z < info.nrImgs; ++z) {
	float sharpness = l_sharpness[x + y*info.w + z*info.w*info.h];
	if (sharpness > max) {
	  max = sharpness;
	  maxIndex = z;
	}
      }

      l_maxValues[x + y*info.w] = max;
      l_indicesMaxValues[x + y*info.w] = maxIndex;
    }
  }
  cout << endl;

  Mat mDepthImg = Mat::zeros(info.h, info.w, CV_32FC1);
  convert_layered_to_mat(mDepthImg, l_indicesMaxValues);  

  return mDepthImg;
}

void LayeredMemory::scaleSharpnessValues(float denomRegu) {
  for(int y = 0; y < info.h; ++y) {
    cout << "\r" << flush;
    cout << "Scale sharpness values of row " << (y+1) << " from " << info.h;              

    for(int x = 0; x < info.w; ++x) {
      float scaling = 1 / (l_maxValues[x + y*info.w] + denomRegu);
      
      for(int z = 0; z < info.nrImgs; ++z) {
	l_sharpness[x + y*info.w + z*info.w*info.h] *=  scaling;
      }
    }
  }
  cout << endl;
}

void LayeredMemory::copySharpnessImageToDevice(float *d_sharpness, size_t idx) {
  cudaMemcpy(d_sharpness, &l_sharpness[idx*info.w*info.h*info.nc], info.w*info.h*info.nc*sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;    
}

void LayeredMemory::copySmoothSharpnessImageToHost(float *d_smooth, size_t idx) {
  cudaMemcpy(&l_sharpness[idx*info.w*info.h*info.nc], d_smooth, info.w*info.h*info.nc*sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;    
}