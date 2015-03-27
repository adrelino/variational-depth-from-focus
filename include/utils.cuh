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
#include <sstream>
#include <iostream>

#include <vector>

// for WINDOWS: has to be downloaded and put into ..\Visual Studio 2013\VC\include or similar
// check: http://www.softagalleria.net/download/dirent/
#include <dirent.h>

#include <sys/types.h>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

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
    // parameter processing
    template<typename T> bool getParam(std::string param, T &var, int argc, char **argv)
    {
      const char *c_param = param.c_str();
      for(int i=argc-1; i>=1; i--)
	{
	  if (argv[i][0]!='-') continue;
	  if (strcmp(argv[i]+1, c_param)==0)
	    {
	      if (!(i+1<argc)) continue;
	      std::stringstream ss;
	      ss << argv[i+1];
	      ss >> var;
	      std::cout<<"PARAM[SET]: "<<param<<" : "<<var<<std::endl;
	      return (bool)ss;
	    }
	}
      std::cout<<"PARAM[DEF]: "<<param<<" : "<<var<<std::endl;
      return false;
    }

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

    std::string getOSSeparator();
    std::vector<std::string> getAllImagesFromFolder(const char *dirname, int skipNthPicture=1);
    float getAverage(const std::vector<float> &v);
    void getAvailableGlobalMemory(size_t *free, size_t *total, bool print=false);
    void memprint();

    char waitKey2(int delay, bool hint=true);
  }
}
#endif

  