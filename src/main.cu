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

#include <cuda.h>
#include <cuda_runtime.h>

#include <utils.cuh>
#include <openCVHelpers.h>

#include <opencv2/core/core.hpp>

#include <LinearizedADMM.cuh>

#include <DataPreparator.cuh>
#include <LayeredMemory.cuh>
#include <DataPreparatorTensor3f.cuh>

#include <CPUTimer.h>
#include <Parameters.h>

using namespace std;
using namespace cv;

using namespace vdff;

bool checkPassedFolderPath(const string path) {
  if (path.empty()) {
    cerr << "You have to deliver a valid (relative or absolute) path to a image sequence." << endl;
    return false;
  }
  return true;
}

// check command line for given parameters; returns false if something is not set correctly
bool parseCmdLine(Parameters &params, int argc, char **argv) {
  Utils::getParam("dir", params.folderPath, argc, argv);
  if (!checkPassedFolderPath(params.folderPath)) {
    return false;
  }

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
  Utils::getParam("convIterations", params.convIterations, argc, argv);
  Utils::getParam("iterations", params.nrIterations, argc, argv);
  Utils::getParam("lambda", params.lambda, argc, argv); 

  Utils::getParam("useNthPicture", params.useNthPicture, argc, argv);
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

  Mat lastImgInSeq = dataLoader->determineSharpnessFromAllImages(params.useNthPicture);
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
  CPUTimer t;
  
  DataPreparator *dataLoader = new LayeredMemory(params.folderPath.c_str(), params.minVal, params.maxVal);
  
  cout << "Determine sharpness from images in " << params.folderPath << endl;
  t.tic();
  dataLoader->determineSharpnessFromAllImages(deviceProperties, params.usePageLockedMemory,
					      params.useNthPicture);
  cudaDeviceSynchronize();
  cout << "time elapsed: " << t.tocInSeconds() << " s" << endl;
  
  cout << endl << "Approximating contrast values" << endl;
  t.tic();
  *d_coefDerivative = dataLoader->calcPolyApproximations(params.polynomialDegree, params.denomRegu);
  cudaDeviceSynchronize();
  cout << "time elapsed: " << t.tocInSeconds() << " s" << endl;

  cout << endl << "Creating smooth depth estimate" << endl;
  t.tic();
  mSmoothDepthEstimateScaled = dataLoader->smoothDepthEstimate(params.smoothGPU);
  cudaDeviceSynchronize();
  cout << "time elapsed: " << t.tocInSeconds() << " s" << endl;
  
  info = dataLoader->getInfoImgSeq();
  return dataLoader;
}

void printGeneralParameters(const Parameters &params) {
  cout << "================================== General Parameters ==============================" << endl;
  cout << "Specified sequence folder: " << params.folderPath << endl;
  cout << "Export filename: " << params.exportFilename << endl;
  cout << "Use page locked memory: " << ((params.usePageLockedMemory) ? "True" : "False") << endl;
  cout << "Use GPU to smooth depth estimate: " << ((params.smoothGPU) ? "True" : "False") << endl;
  cout << "Use every n-th picture: " << params.useNthPicture << endl;
  cout << "Use Tensor3f Class (experimental): " << ((params.useTensor3fClass) ? "True" : "False") << endl;
  cout << "Delay: " << params.delay << endl;
  cout << "minVal: " << params.minVal << endl;
  cout << "maxVal: " << params.maxVal << endl;
  cout << "Degree of polynomial: " << params.polynomialDegree << endl;
  cout << "DenomRegu: " << params.denomRegu << endl;
  cout << "====================================================================================" << endl;
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
  printGeneralParameters(params);

  cout << "\n================================== Executing =======================================" << endl;
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

  // Tensor3f is a convenience class, which is still in an experimental status; it does for example
  // not yet work for larger datasets. Thus, right now, it should not be used.
  if (params.useTensor3fClass) {
    dataLoaderTensor3f = approximateSharpnessAndCreateDepthEstimateTensor3f(params, deviceProperties,
									    &d_coefDerivative, mSmoothDepthEstimateScaled, info);
  }
  else {
    dataLoader = approximateSharpnessAndCreateDepthEstimate(params, deviceProperties,
							    &d_coefDerivative, mSmoothDepthEstimateScaled, info);
  }

  cout << endl << "Running ADMM" << endl;
  cout << "Parameters:" << endl;
  cout << "\tDataFidelityParam: " << params.dataFidelityParam << endl;
  cout << "\tDataDescentStep: " << params.dataDescentStep << endl;
  cout << "\tconvIterations: " << params.convIterations << endl;
  cout << "\tnrIterations: " << params.nrIterations << endl;
  cout << "\tlambda: " << params.lambda << endl;
  cout << endl;

  methods->tic();
  LinearizedADMM admm(mSmoothDepthEstimateScaled.cols, mSmoothDepthEstimateScaled.rows,
		      params.minVal, params.maxVal);
  
  Mat res = admm.run(d_coefDerivative, params.polynomialDegree, params.dataFidelityParam,
		     params.dataDescentStep, mSmoothDepthEstimateScaled, false,
		     params.convIterations, params.nrIterations, params.lambda);

  cudaDeviceSynchronize(); CUDA_CHECK;
  cout << "time elapsed: " << methods->tocInSeconds() << " s" << endl; 
  cout << "======================================================================" <<endl;
  cout << "Total elapsed time: " << total->tocInSeconds() << " s" << endl;

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
