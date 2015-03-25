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
#include <helper.h>
#include <iostream>
#include <openCVHelpers.h>
#include <cstring>
#include <opencv2/contrib/contrib.hpp>

#include <cuda.h>
#include <stdio.h>

using namespace std;
using namespace cv;

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

void imagesc(std::string title, cv::Mat mat, int x, int y) {
  double min,max;
  cv::minMaxLoc(mat,&min,&max);

  Mat scaled = mat;
  Mat meanCols;
  reduce(mat, meanCols, 0, CV_REDUCE_AVG );

  Mat mean;
  reduce(meanCols, mean, 1, CV_REDUCE_AVG);
    
  cout << "Max value: " << max << endl;
  cout << "Mean value: " << mean.at<float>(0) << endl;
  cout << "Min value: " << min << endl;
    
  if (std::abs(max) > 0.0000001f)
    scaled /= max;

  showImage(title, scaled, x,y);
}

char waitKey2(int delay, bool hint){
  char c;
  if(hint){
    cout << "delay="<<delay<<endl;
    if(delay < 0){
      cout<<"[CONSOLE]: press key to continue"<<endl;
    }else if (delay == 0) {
      cout<<"[OpenCV WINDOW]: press key to continue"<<endl;
    }else{
      cout<<"[GENERAL]: waiting for "<< delay <<" ms"<<endl;
    }
  }
  int wait=delay;
  if(wait<0) wait*=-1;
  c=waitKey(wait);
  if(delay<0){
    std::string input;
    std::getline(std::cin,input);
    c=*input.c_str();
  }
  return c;  
}

void createOptimallyPaddedImageForDCT(const Mat& img, Mat& paddedImg, 
				      int &paddingX, int &paddingY) {
  // pad init if it is not divisible by 2
  int maxVecSize = max(img.rows, img.cols);
  int optVecSize = getOptimalDFTSize((maxVecSize+1)/2)*2;

  paddingX = optVecSize - img.cols;
  paddingY = optVecSize - img.rows;

  int top, bottom, left, right;
  top = bottom = paddingY / 2;
  left = right = paddingX / 2;
  bottom += paddingY % 2 == 1;  
  right += paddingX % 2 == 1;

  if (paddingX == 0 && paddingY == 0) {
    paddedImg = img.clone();
  }
  else {
    copyMakeBorder(img, paddedImg, top, bottom, left, right, BORDER_CONSTANT, Scalar(0));
  }  
}

void showDepthImage(const string &wndTitle, const Mat& img, int posX, int posY, bool doResize) {
  double min, max;
  minMaxIdx(img, &min, &max);

  Mat depthMap;
  float scale = 255.0f / (max - min);
  img.convertTo(depthMap, CV_8UC1, scale, -min*scale);

  Mat heatMap;
  applyColorMap(depthMap, heatMap, cv::COLORMAP_JET);

  if (doResize)
    resize(heatMap, heatMap, Size(), 0.5, 0.5);
  
  showImage(wndTitle, heatMap, posX, posY);
}

string getOSSeparator() {
#ifdef _WIN32
  return "\\";
#else
  return "/";
#endif
}

vector<string> getAllImagesFromFolder(const char *dirname) {
  DIR *dir = NULL;
  struct dirent *entry;
  vector<string> allImages;

  dir = opendir(dirname);

  if (!dir) {
    cerr << "Could not open directory " << dirname << ". Exiting..." << endl;
    exit(1);
  }
  
  const string sep = getOSSeparator();
  string dirStr = string(dirname);

  while(entry = readdir(dir)) {
    if (strstr(entry->d_name, ".png") ||
	strstr(entry->d_name, ".jpg") ||
	strstr(entry->d_name, ".tif")) {
      string fileName(entry->d_name);
      string fullPath = dirStr + sep + fileName;
      allImages.push_back(fullPath);
    }
  }
  closedir(dir);

  // sort string alphabetically
  std::sort(allImages.begin(), allImages.end());
  return allImages;
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

void printSharpnessValues(float *l_sharpness, size_t x, size_t y, size_t w, size_t h, size_t n) {
  cout << "Sharpness values at (y: " << y << ", x: " << x << "): ";
  for(size_t i = 0; i < n; ++i) {
    cout << l_sharpness[x + y*w + i*w*h] << ", ";
  }
  cout << endl;
}

// degree is polynomial degree!
void printCoefficients(float *l_coef, size_t x, size_t y, size_t w, size_t h, size_t degree) {
  cout << "Coefficient at (y: " << y << ", x: " << x << "): ";
  for(size_t i = 0; i < degree+1; ++i) {
    cout << l_coef[x + y*w + i*w*h] << ", ";
  }
  cout << endl;
}

// degree is polynomial degree! (not degree of the Derivatives.)
void printDerivativeCoefficients(float *l_coefDeriv, size_t x, size_t y, size_t w, size_t h, size_t degree) {
  cout << "Derivative Coefficient at (y: " << y << ", x: " << x << "): ";
  for(size_t i = 0; i < degree; ++i) {
    cout << l_coefDeriv[x + y*w + i*w*h] << ", ";    
  }
  cout << endl;
}