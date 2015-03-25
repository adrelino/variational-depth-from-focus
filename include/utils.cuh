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
#include <vector>

// for WINDOWS: has to be downloaded and put into ..\Visual Studio 2013\VC\include or similar
// check: http://www.softagalleria.net/download/dirent/
#include <dirent.h>

#include <sys/types.h>
#include <vector>

#include <opencv2/core/core.hpp>

#include <CUDATimer.h>
#include <stdio.h>

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
void imagesc(std::string title, cv::Mat mat, int x, int y);
void createOptimallyPaddedImageForDCT(const cv::Mat& img, cv::Mat& paddedImg, 
				      int &paddingX, int &paddingY);
void showDepthImage(const std::string &wndTitle, const cv::Mat& img, int posX, int posY, bool dResize=false);
std::string getOSSeparator();
std::vector<std::string> getAllImagesFromFolder(const char *dirname);
float getAverage(const std::vector<float> &v);
void getAvailableGlobalMemory(size_t *free, size_t *total, bool print=false);
void memprint();

char waitKey2(int delay, bool hint=true);

void printSharpnessValues(float *l_coef, size_t x, size_t y, size_t w, size_t h, size_t n);
void printCoefficients(float *l_coef, size_t x, size_t y, size_t w, size_t h, size_t degree);
void printDerivativeCoefficients(float *l_coefDeriv, size_t x, size_t y, size_t w, size_t h, size_t degree);

#endif
