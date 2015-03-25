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
#include <iostream>
#include <cstdio>
#include <stdlib.h>
#include <vector>
#include <string>
#include <algorithm>
#include <limits>

// cuda stuff
#include <cuda.h>
#include <cuda_runtime.h>

#include <helper.h>
#include <common_kernels.cuh>
#include <LaplaceInversion.cuh>
#include <openCVHelpers.h>
#include <CUDATimer.h>
#include <cudaWrappers.h>

// opencv stuff
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>

// util functions for main.cu
#include <utils.cuh>
#include <LinearizedADMM.cuh>
#include <DataPreparator.cuh>
#include <LayeredMemory.cuh>
#include <DataPreparatorTensor3f.cuh>

// using clock_gettime
#include <CPUTimer.h>

using namespace std;
using namespace cv;

// general variables
cudaDeviceProp deviceProperties;

// absolute path to the image sequence
string folderPath = "../samples/sim";

// [minVal, maxVal] determines in which range we will do the
// polynomial approximation later
float minVal = -10.0f;
float maxVal = 10.0f;
size_t polynomialDegree = 6;

// value to decrease the importance of sharp edges
float denomRegu = 0.3f; //0.8f no holes

float dataFidelityParam = 6.0f;  //2.5f no holes
float dataDescentStep = 8.0f / dataFidelityParam;
// show iteration steps?
bool plotIterations = false;
// nr iterations for ADMM algorithm
size_t convIterations = 0;
size_t nrIterations = 400;
// lambda in ADMM procedure
float lambda = 1.0f;
int delay = 1;

bool useTensor3fClass = false;

bool usePageLockedMemory = false;
bool smoothGPU = true;

// TODO: more error checking
void checkPassedFolderPath(const string path) {
  if (path.empty()) {
    cerr << "You have to deliver a valid (relative or absolute) path to a image sequence." << endl;
    exit(1);
  }
}

void parseCmdLine(int argc, char **argv) {
  getParam("dir", folderPath, argc, argv);
  getParam("useTensor3fClass", useTensor3fClass, argc, argv);
  getParam("delay", delay, argc, argv);

  getParam("smoothGPU", smoothGPU, argc, argv);
  getParam("pageLocked", usePageLockedMemory, argc, argv);

  checkPassedFolderPath(folderPath);

  getParam("minVal", minVal, argc, argv);
  getParam("maxVal", maxVal, argc, argv);
  getParam("denomRegu", denomRegu, argc, argv);
  getParam("polyDegree", polynomialDegree, argc, argv);

  // if dataFidelity was set, make sure tau(dataDescentStep) stays up2date
  getParam("dataFidelity", dataFidelityParam, argc, argv);
  dataDescentStep = 8.0f / dataFidelityParam;

  getParam("descentStep", dataDescentStep, argc, argv);

  getParam("plotIterations", plotIterations, argc, argv);
  getParam("convIterations", convIterations, argc, argv);
  getParam("iterations", nrIterations, argc, argv);
  getParam("lambda", lambda, argc, argv);  
}

void wait(){
  waitKey2(delay);
}

int main(int argc, char **argv) {
  CPUTimer *total, *methods;
  total = new CPUTimer();
  methods = new CPUTimer();

  // read in arguments from command-line
  parseCmdLine(argc, argv);

  cudaDeviceReset(); CUDA_CHECK;

  // initialize CUDA context
  cudaDeviceSynchronize(); CUDA_CHECK;
  
  total->tic();
  deviceProperties = queryDeviceProperties();

  size_t freeStartup, totalStartup;
  getAvailableGlobalMemory(&freeStartup, &totalStartup);

  Mat mSmoothDepthEstimateScaled; //interface to pass to ADMM after our 2 different loading classes
  float *d_coefDerivative;
  InfoImgSeq info;

  if(useTensor3fClass){

    DataPreparatorTensor3f *dataLoader = new DataPreparatorTensor3f(folderPath.c_str(), minVal, maxVal);

    Mat lastImgInSeq = dataLoader->determineSharpnessFromAllImages();
    showImage("Last Image in Sequence", lastImgInSeq, 50, 50); wait();

    int lastIndex=dataLoader->getInfoImgSeq().nrImgs - 1;


    dataLoader->t_sharpness->download();
    showImage("Last Image Sharpness Measure", dataLoader->t_sharpness->getImageInSequence(lastIndex), 100, 100); wait();


    dataLoader->findMaxSharpnessValues();
    dataLoader->t_noisyDepthEstimate->download();
    showDepthImage("Noisy Depth Estimate", dataLoader->t_noisyDepthEstimate->getMat(), 150, 150); wait();
    
    delete dataLoader->t_noisyDepthEstimate;


    dataLoader->scaleSharpnessValues(denomRegu);



    dataLoader->approximateContrastValuesWithPolynomial(polynomialDegree);


    dataLoader->smoothDepthEstimate();

    //pass on interface imgs
    d_coefDerivative = dataLoader->t_coefDerivative->getDevicePtr();
    mSmoothDepthEstimateScaled = dataLoader->smoothDepthEstimate_ScaleIntoPolynomialRange();
    //end interface

    showDepthImage("Smoothed Depth Estimate", mSmoothDepthEstimateScaled, 200, 200); wait();

    delete dataLoader->t_sharpness;

    //from here on, we only need the following, so dont free them yet:
    //Tensor3f *t_coefDerivative;
    //Tensor3f *t_smoothDepthEstimate; --> actually now, we save intermediate result on cpu, so can delete
    delete dataLoader->t_smoothDepthEstimate;

    info=dataLoader->getInfoImgSeq();
  }else{

    DataPreparator *dataLoader = new LayeredMemory(folderPath.c_str(), minVal, maxVal);
    
    cout << "Determine sharpness from images in " << folderPath << endl;
    methods->tic();
    dataLoader->determineSharpnessFromAllImages(deviceProperties, usePageLockedMemory);
    cudaDeviceSynchronize();
    methods->toc("1-determineSharpness");

    cout << "Approximating contrast values" << endl;
    methods->tic();
    d_coefDerivative = dataLoader->calcPolyApproximations(polynomialDegree, denomRegu);
    cudaDeviceSynchronize();
    methods->toc("2-approxContrast");

    methods->tic();
    mSmoothDepthEstimateScaled = dataLoader->smoothDepthEstimate(smoothGPU);
    cudaDeviceSynchronize();
    methods->toc("3-depthInit");

    info=dataLoader->getInfoImgSeq();
  }

  cout << "============================== Parameters for ADMM ================================================" << endl;
  cout << "DataFidelityParam: " << dataFidelityParam << endl;
  cout << "DataDescentStep: " << dataDescentStep << endl;
  cout << "plotIterations: " << (plotIterations ? "True" : "False") << endl;
  cout << "convIterations: " << convIterations << endl;
  cout << "nrIterations: " << nrIterations << endl;
  cout << "lambda: " << lambda << endl;
  
  cout << "--------------------------------  Starting ADMM  --------------------------------------------------" << endl;
  

  methods->tic();
  LinearizedADMM admm(mSmoothDepthEstimateScaled.cols, mSmoothDepthEstimateScaled.rows, minVal, maxVal);
  
  Mat res = admm.run(d_coefDerivative, polynomialDegree, dataFidelityParam, dataDescentStep,
  		     mSmoothDepthEstimateScaled, plotIterations, convIterations, nrIterations, lambda);

  cudaDeviceSynchronize(); CUDA_CHECK;

  methods->toc("4-timingADMM"); 

  cout << "====================================================================================================" << endl;  

  total->toc("5-Total");

  cout<<"======ALLTimings======="<<endl;
  cout<<"Dir:          "<<folderPath<<endl;
  info.print();
  methods->printAllTimings();
  total->printAllTimings();

  showDepthImage("3-smoothed Depth Estimate", mSmoothDepthEstimateScaled, 500, 100, true);wait();  
  showDepthImage("Result", res, 250 , 250); 

  //require user input to exit
  waitKey(0); 
  return 0;
}
