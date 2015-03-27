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
#include <iostream>
#include <openCVHelpers.h>
#include <cstring>

#include <cuda.h>
#include <stdio.h>

#ifndef __USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#include <math.h>
#endif
#include <cstdlib>

using namespace std;
using namespace cv;

// parameter processing: template specialization for T=bool
template<> inline bool getParam<bool>(std::string param, bool &var, int argc, char **argv)
{
    const char *c_param = param.c_str();
    for(int i=argc-1; i>=1; i--)
    {
        if (argv[i][0]!='-') continue;
        if (strcmp(argv[i]+1, c_param)==0)
        {
            if (!(i+1<argc) || argv[i+1][0]=='-') { var = true; return true; }
            std::stringstream ss;
            ss << argv[i+1];
            ss >> var;
            return (bool)ss;
        }
    }
    return false;
}

// opencv helpers


// cuda error checking
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



string getOSSeparator() {
#ifdef _WIN32
  return "\\";
#else
  return "/";
#endif
}

vector<string> getAllImagesFromFolder(const char *dirname, int skipNthPicture) {
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

  // delete some pictures if desired
  if (skipNthPicture > 1) {

    // some sanity check
    if (skipNthPicture >= allImages.size()) {
      cerr << "You can not skip " << skipNthPicture << " since there are only " << allImages.size() 
  	   << " pictures in your chosen folder.\nPlease adjust your parameter." << endl;
      exit(1);
    }
    
    vector<string> reduced;
    for (size_t i = 0; i < allImages.size(); ++i) {
      if ((i % skipNthPicture) == 0)
	continue;
      
      reduced.push_back(allImages.at(i));
    }
    return reduced;
  }
  else {
    return allImages;
  }
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