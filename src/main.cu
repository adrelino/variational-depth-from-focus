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

#include <string>
#include <algorithm>
#include <cctype>

// cuda stuff
#include <cuda.h>
#include <cuda_runtime.h>

#include <utils.cuh>
#include <common_kernels.cuh>
#include <LaplaceInversion.cuh>
#include <openCVHelpers.h>
#include <CUDATimer.h>
#include <cudaWrappers.h>

// opencv stuff
#include <opencv2/core/core.hpp>

// util functions for main.cu
#include <utils.cuh>
#include <LinearizedADMM.cuh>
#include <DataPreparator.cuh>
#include <LayeredMemory.cuh>
#include <DataPreparatorTensor3f.cuh>

// using clock_gettime
#include <CPUTimer.h>

#include <Parameters.h>

using namespace std;
using namespace cv;

using namespace vdff;

// PARAMETERS --------------------------------------------------
// general variables

// absolute path to the image sequence
// string folderPath = "../samples/sim";

// // [minVal, maxVal] determines in which range we will do the
// // polynomial approximation later
// float minVal = -10.0f;
// float maxVal = 10.0f;
// size_t polynomialDegree = 6;

// // value to decrease the importance of sharp edges
// float denomRegu = 0.3f; //0.8f no holes

// float dataFidelityParam = 6.0f;  //2.5f no holes
// float dataDescentStep = 8.0f / dataFidelityParam;
// // show iteration steps?
// bool plotIterations = false;
// // nr iterations for ADMM algorithm
// size_t convIterations = 0;
// size_t nrIterations = 400;
// // lambda in ADMM procedure
// float lambda = 1.0f;
// int delay = 1;

// bool useTensor3fClass = false;

// bool usePageLockedMemory = false;
// bool smoothGPU = true;

// int skipNthPicture = 1;
// string exportFilename = "";
//----------------------------------------------------------------------------------------------------

bool checkPassedFolderPath(const string path) {
  if (path.empty()) {
    cerr << "You have to deliver a valid (relative or absolute) path to a image sequence." << endl;
    return false;
  }
  return true;
}

bool parseCmdLine(Parameters &params, int argc, char **argv) {
  Utils::getParam("dir", params.folderPath, argc, argv);
  checkPassedFolderPath(params.folderPath);

  Utils::getParam("useTensor3fClass", params.useTensor3fClass, argc, argv);
  Utils::getParam("delay", params.delay, argc, argv);

  Utils::getParam("smoothGPU", params.smoothGPU, argc, argv);
  Utils::getParam("pageLocked", params.usePageLockedMemory, argc, argv);

  Utils::getParam("minVal", params.minVal, argc, argv);
  Utils::getParam("maxVal", params.maxVal, argc, argv);
  Utils::getParam("denomRegu", params.denomRegu, argc, argv);
  Utils::getParam("polyDegree", params.polynomialDegree, argc, argv);

  // if dataFidelity was set, make sure tau(dataDescentStep) stays up2date
  Utils::getParam("dataFidelity", params.dataFidelityParam, argc, argv);
  params.dataDescentStep = 8.0f / params.dataFidelityParam;

  Utils::getParam("descentStep", params.dataDescentStep, argc, argv);

  Utils::getParam("plotIterations", params.plotIterations, argc, argv);
  Utils::getParam("convIterations", params.convIterations, argc, argv);
  Utils::getParam("iterations", params.nrIterations, argc, argv);
  Utils::getParam("lambda", params.lambda, argc, argv); 

  Utils::getParam("skipNthPicture", params.skipNthPicture, argc, argv);
  Utils::getParam("export", params.exportFilename, argc, argv);

  return true;
}

// wait delay seconds
void wait(int delay){
  Utils::waitKey2(delay);
}

DataPreparatorTensor3f* approximateSharpnessAndCreateDepthEstimateTensor3f(const Parameters &params,
							const cudaDeviceProp &deviceProperties,
							float **d_coefDerivative,
							Mat &mSmoothDepthEstimateScaled,
							Utils::InfoImgSeq &info) {
  DataPreparatorTensor3f *dataLoader = new DataPreparatorTensor3f(params.folderPath.c_str(), params.minVal, params.maxVal);

  Mat lastImgInSeq = dataLoader->determineSharpnessFromAllImages(params.skipNthPicture);
  int lastIndex=dataLoader->getInfoImgSeq().nrImgs - 1;


  dataLoader->t_sharpness->download();

  dataLoader->findMaxSharpnessValues();
  dataLoader->t_noisyDepthEstimate->download();
    
  delete dataLoader->t_noisyDepthEstimate;
  dataLoader->t_noisyDepthEstimate = NULL;
  
  dataLoader->scaleSharpnessValues(params.denomRegu);
  dataLoader->approximateContrastValuesWithPolynomial(params.polynomialDegree);
  dataLoader->smoothDepthEstimate();

  //pass on interface imgs
  *d_coefDerivative = dataLoader->t_coefDerivative->getDevicePtr();
  mSmoothDepthEstimateScaled = dataLoader->smoothDepthEstimate_ScaleIntoPolynomialRange();
  //end interface

  delete dataLoader->t_sharpness;
  // do not forget to set pointer to NULL
  dataLoader->t_sharpness = NULL;

  //from here on, we only need the following, so dont free them yet:
  delete dataLoader->t_smoothDepthEstimate;
  dataLoader->t_smoothDepthEstimate = NULL;
  
  info = dataLoader->getInfoImgSeq();    
  return dataLoader;
}

DataPreparator* approximateSharpnessAndCreateDepthEstimate(const Parameters &params,
							   const cudaDeviceProp &deviceProperties,
							   float **d_coefDerivative,
							   Mat &mSmoothDepthEstimateScaled,
							   Utils::InfoImgSeq &info) {
  DataPreparator *dataLoader = new LayeredMemory(params.folderPath.c_str(), params.minVal, params.maxVal);
  
  cout << "Determine sharpness from images in " << params.folderPath << endl;
  dataLoader->determineSharpnessFromAllImages(deviceProperties, params.usePageLockedMemory,
  						params.skipNthPicture);
  cudaDeviceSynchronize();

  cout << "Approximating contrast values" << endl;
  *d_coefDerivative = dataLoader->calcPolyApproximations(params.polynomialDegree, params.denomRegu);
  cudaDeviceSynchronize();

  mSmoothDepthEstimateScaled = dataLoader->smoothDepthEstimate(params.smoothGPU);
  cudaDeviceSynchronize();

  info = dataLoader->getInfoImgSeq();
  return dataLoader;
}

int main(int argc, char **argv) {
  CPUTimer *total, *methods;
  total = new CPUTimer();
  methods = new CPUTimer();

  // read in arguments from command-line
  Parameters params;
  bool succ = parseCmdLine(params, argc, argv);
  if (!succ) {
    cerr << "Problem in parsing arguments. Aborting..." << endl;
    exit(1);
  }

  cudaDeviceReset(); CUDA_CHECK;

  // initialize CUDA context
  cudaDeviceSynchronize(); CUDA_CHECK;
  
  total->tic();
  cudaDeviceProp deviceProperties = Utils::queryDeviceProperties();

  size_t freeStartup, totalStartup;
  Utils::getAvailableGlobalMemory(&freeStartup, &totalStartup);

  Mat mSmoothDepthEstimateScaled; //interface to pass to ADMM after our 2 different loading classes
  float *d_coefDerivative = NULL; // will contain the coefficients of the polynomial for each pixel
  Utils::InfoImgSeq info;

  DataPreparator *dataLoader = NULL;
  DataPreparatorTensor3f *dataLoaderTensor3f = NULL;
  
  // load all images in the given folderPath, determine the sharpness of each image
  // and then approximate the sharpness values via polynomials for each pixel.
  // Additionally, create a smooth depth estimate, which serves as a starting point for the
  // ADMM algorithm
  if (params.useTensor3fClass) {
    dataLoaderTensor3f = approximateSharpnessAndCreateDepthEstimateTensor3f(params, deviceProperties,
									    &d_coefDerivative, mSmoothDepthEstimateScaled, info);
  }
  else {
    dataLoader = approximateSharpnessAndCreateDepthEstimate(params, deviceProperties,
							    &d_coefDerivative, mSmoothDepthEstimateScaled, info);
  }

  cout << "============================== Parameters for ADMM ================================================" << endl;
  cout << "DataFidelityParam: " << params.dataFidelityParam << endl;
  cout << "DataDescentStep: " << params.dataDescentStep << endl;
  cout << "plotIterations: " << (params.plotIterations ? "True" : "False") << endl;
  cout << "convIterations: " << params.convIterations << endl;
  cout << "nrIterations: " << params.nrIterations << endl;
  cout << "lambda: " << params.lambda << endl;
  
  cout << "--------------------------------  Starting ADMM  --------------------------------------------------" << endl;
  
  methods->tic();
  LinearizedADMM admm(mSmoothDepthEstimateScaled.cols, mSmoothDepthEstimateScaled.rows,
  		      params.minVal, params.maxVal);
  
  Mat res = admm.run(d_coefDerivative, params.polynomialDegree, params.dataFidelityParam,
  		     params.dataDescentStep, mSmoothDepthEstimateScaled, params.plotIterations,
  		     params.convIterations, params.nrIterations, params.lambda);

  cudaDeviceSynchronize(); CUDA_CHECK;

  methods->toc("4-timingADMM"); 

  cout << "====================================================================================================" << endl;  

  total->toc("5-Total");

  cout<<"======ALLTimings======="<<endl;
  cout<<"Dir:          "<< params.folderPath<<endl;
  info.print();
  methods->printAllTimings();
  total->printAllTimings();

  Mat resHeatMap = openCVHelpers::showDepthImage("Result", res, 250 , 250);
  //require user input to exit
  waitKey(0);

  // if user specified an export file, we save the result
  if(!params.exportFilename.empty()) {
    openCVHelpers::exportImage(resHeatMap, params.exportFilename);
  }

  if (dataLoaderTensor3f) {
    delete dataLoaderTensor3f;
  }

  if (dataLoader) {
    delete dataLoader;
  }
  return 0;
}
