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

#include <DataPreparatorTensor3f.cuh>

#include <opencv2/core/core.hpp>
#include <cuda.h>
#include <cudaWrappers.h>
#include <iostream>
#include <utils.cuh>

using namespace std;
using namespace cv;

namespace vdff {
  DataPreparatorTensor3f::DataPreparatorTensor3f(const char *dir, float min, float max) 
    : dirPath(dir), scale(max - min), findMaxGPU(true), scaleSharpnessGPU(true)
  {
    methods = new CPUTimer();

    printf("[CONSTRUCTOR] DataPreparatorTensor3f: dir=%s\n",dir);
    printf("[     ||    ]       ||      : scale=%f findMaxGPU=%d scaleSharpnessGPU=%d\n",scale,findMaxGPU,scaleSharpnessGPU);

  }

  DataPreparatorTensor3f::~DataPreparatorTensor3f() {
    delete methods;

    if(t_imgSeq != 0){
      delete t_imgSeq;
    }
    
    if(t_sharpness != 0){
      delete t_sharpness;
    }
    if(t_noisyDepthEstimate != 0){
      delete t_noisyDepthEstimate;
    }
    if(t_maxValues != 0){
      delete t_maxValues;
    }
    if (t_coefDerivative != 0) {
      delete t_coefDerivative;
    }
  }

  Mat DataPreparatorTensor3f::determineSharpnessFromAllImages(int skipNthPicture) {
    //in: nothing
    cout<<"DataPreparatorTensor3f::determineSharpnessFromAllImages"<<endl;
    methods->tic();

    vector<string> imgFileNames = Utils::getAllImagesFromFolder(dirPath);

    size_t nrImgs = imgFileNames.size();
    string imgFile = imgFileNames[0];

    const int imgLoadFlag = CV_LOAD_IMAGE_ANYDEPTH;  

    Mat curImg = imread(imgFileNames[0], imgLoadFlag);
    int w = curImg.cols;
    int h = curImg.rows;
    int nc = curImg.channels();

    // determine here if we need a padding
    // since we are assuming all images have same size!
    int optW = getOptimalDFTSize((w+1)/2)*2;
    int optH = getOptimalDFTSize((h+1)/2)*2;

    int diffW = optW - w;
    int diffH = optH - h;

    int paddingTop, paddingBottom, paddingLeft, paddingRight;
    paddingTop = paddingBottom = diffH / 2;
    paddingLeft = paddingRight = diffW / 2;
    paddingBottom += diffH % 2 == 1;
    paddingRight += diffW % 2 == 1;

    if (diffW != 0 || diffH != 0)
      copyMakeBorder(curImg, curImg, paddingTop, paddingBottom, paddingLeft, paddingRight, BORDER_REPLICATE);
  
    //showImage("now with border", curImg, 50, 50); waitKey(0);

    // fill info struct //stored in class, can be reused later
    info.w = curImg.cols;
    info.h = curImg.rows;
    info.nrImgs = nrImgs;
    // ATTENTION: we have only one channel in the sharpness image!
    info.nc = 1;

    size_t freeMemory, totalMemory;

    Utils::getAvailableGlobalMemory(&freeMemory, &totalMemory, true);

    //Allocate output 3D cube (of size input / nc)
    t_sharpness = new Tensor3f(w,h,1,nrImgs,"sharpness");
    printf("OK\n");

    Utils::getAvailableGlobalMemory(&freeMemory, &totalMemory,true);

    //Allocate input 3D cube
    t_imgSeq = new Tensor3f(w,h,nc,nrImgs,"imgSeq");             
    printf("OK\n");

    Utils::getAvailableGlobalMemory(&freeMemory, &totalMemory,true);

    cout << endl;

    for(int i = 0; i < nrImgs; ++i) {
    
      if (skipNthPicture > 1) {
	if ((i % skipNthPicture) == 0) {
	  continue;
	}
      }
    
      cout << "\r" << flush;
      cout << "Loading picture into Host Memory: " << (i+1) << " from " << nrImgs;

      if (i != 0) {
	imgFile = imgFileNames[i];
	//cout<<"before reading: "<<imgFile<<endl;
	curImg = imread(imgFile, imgLoadFlag);
      }
      //cout<<"before converting"<<endl;
      curImg.convertTo(curImg, CV_32F, 1.0f/255.0f);

      // check if we got the same size, before the (possible) padding!
      assert(w == curImg.cols && h == curImg.rows && nc == curImg.channels());

      // pad the image if we need to, to guarantee that DCT is possible
      if (diffW != 0 || diffH != 0)
	copyMakeBorder(curImg, curImg, paddingTop, paddingBottom, paddingLeft, paddingRight, BORDER_REPLICATE);

      //cout<<"before setting img in sequence"<<endl;
      t_imgSeq->setImageInSequence(i,curImg);
    }
  
    blockSize = dim3(32, 8, 1);  //class variables, reused in other kernel launches
    gridSize = dim3((w + blockSize.x - 1) / blockSize.x, (h + blockSize.y - 1) / blockSize.y, 1);

    cout<<"launching MLAP kernel"<<endl;
    MLAP<<<gridSize, blockSize>>>(t_imgSeq->upload(), t_sharpness->getDevicePtrAllocated(), w, h, nc, nrImgs); CUDA_CHECK;
    cudaDeviceSynchronize(); CUDA_CHECK;
    cout<<"t_sharpness now on Device"<<endl;

    cout << endl;

    delete t_imgSeq; //saves memory, but if we load (TODO) tiled for big images, we need to keep at least the host part of this
    t_imgSeq = NULL;

    methods->toc("determineSharpnessFromAllImages");

    return curImg;
    //out: imgSeq, sharpness
  }

  void DataPreparatorTensor3f::findMaxSharpnessValues() {
    //in: sharpness

    cout<<"DataPreparatorTensor3f::findMaxSharpnessValues findMaxGPU="<<findMaxGPU<<endl;
    methods->tic();

    //allocate mem
    t_maxValues = new Tensor3f(info.w, info.h,1,1,"maxValues");
    t_noisyDepthEstimate = new Tensor3f(info.w, info.h,1,1,"noisyDepthEstimate");
  
    if(findMaxGPU){

      cudaFindMax(gridSize,blockSize,t_sharpness->getDevicePtr(),t_maxValues->getDevicePtrAllocated(),t_noisyDepthEstimate->getDevicePtrAllocated(),info.w,info.h,info.nrImgs);
      cudaDeviceSynchronize(); CUDA_CHECK;

    }else{

      const float* l_sharpness = t_sharpness->download();
      float* l_maxValues = t_maxValues->getHostPtrAllocated();
      float* l_indicesMaxValues = t_noisyDepthEstimate->getHostPtrAllocated();
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
      t_maxValues->upload();//make also available on GPU
      t_noisyDepthEstimate->upload();
    }

    methods->toc("findMaxSharpnessValues");

    //out: noisyDepthEstimate, maxValues
  }

  void DataPreparatorTensor3f::scaleSharpnessValues(float denomRegu) {
    //in: sharpness, maxValues

    cout<<"DataPreparatorTensor3f::scaleSharpnessValues with denomRegu="<<denomRegu<<" scaleSharpnessGPU="<<scaleSharpnessGPU<<endl;
    methods->tic();

    if(scaleSharpnessGPU){
      cudaScaleSharpnessValues(gridSize,blockSize,t_maxValues->getDevicePtr(),t_sharpness->getDevicePtr(),info.w,info.h,info.nrImgs,denomRegu);
      cudaDeviceSynchronize(); CUDA_CHECK;
    }else{

      float* l_sharpness=t_sharpness->download(); //by default, we only store on gpu
      float* l_maxValues=t_maxValues->download();
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
      t_sharpness->upload(); // each kernel is responsible that data is present on gpu
    }

    methods->toc("scaleSharpnessValues");

    //out: sharpness (overwritten, not newly allocated)
  }

  Mat DataPreparatorTensor3f::precomputePseudoInverse(const float *xi, size_t nrUnknowns, size_t polynomialDegree) {
    Mat X = Mat::ones(nrUnknowns, polynomialDegree+1, CV_32FC1);
  
    for (size_t i =0 ; i < nrUnknowns; ++i) {
      float curXi = xi[i];
      for (int j = X.cols-2; j >= 0; --j) {
	X.at<float>(i, j) = curXi;
	curXi *= xi[i];
      }
    }

    Mat tmp = X.t() * X;
    Mat tmpInverse = tmp.inv();
    return tmpInverse*X.t();
  }

  void DataPreparatorTensor3f::approximateContrastValuesWithPolynomial(size_t degree) {
    //in: sharpness

    cout << "DataPreparatorTensor3f::approximateContrastValuesWithPolynomial degree=" << degree << endl;
    methods->tic();

    float *xi = new float[info.nrImgs];
    for(int i = 0; i < info.nrImgs; ++i) {
      xi[i] = scale * ((i / static_cast<float>(info.nrImgs-1)) - 0.5f);
    }
    // compute pseudo-inverse
    Mat mpInv = precomputePseudoInverse(xi, info.nrImgs, degree);
    delete[] xi;

    Tensor3f t_mpInv(mpInv,"mpInv-STACK");

    // allocate stuff for coefficients and derivatives of the coefficients
    // will later be the pointer for d_coef (after we multiply with -1)
    Tensor3f t_negCoef(info.w,info.h,1,degree+1,"negCoef-STACK");
  
    cout << "polyfit" << endl;
    cudaPolyfit(gridSize, blockSize,t_mpInv.upload(), t_sharpness->getDevicePtr(), t_negCoef.getDevicePtrAllocated(), info.w, info.h, info.nrImgs,degree+1); CUDA_CHECK;    
    cudaDeviceSynchronize(); CUDA_CHECK;


    t_coefDerivative = new Tensor3f(info.w,info.h,1,degree,"coefDerivative"); //dynamic alloc, since used later

    cout << "polyder" << endl; 
    cudaPolyder(gridSize, blockSize, t_negCoef.getDevicePtr(), t_coefDerivative->getDevicePtrAllocated(), info.w, info.h, degree+1); CUDA_CHECK;
    cudaDeviceSynchronize(); CUDA_CHECK;
  
    // negate the derivatives
    cout << "cudaMultiplyArrayWithScalar" << endl;
    cudaMultiplyArrayWithScalar(gridSize, blockSize, t_coefDerivative->getDevicePtr(), -1.0f, t_coefDerivative->getDevicePtr(), info.w, info.h, degree); CUDA_CHECK;
    cudaDeviceSynchronize(); CUDA_CHECK;

    t_mpInv.free();
    t_negCoef.free();

    methods->toc("approximateContrastValuesWithPolynomial");

    //out: coefDerivative
  }

  void DataPreparatorTensor3f::smoothDepthEstimate() {
    //in: sharpness

    cout << "DataPreparatorTensor3f::smoothDepthEstimate"<<endl;
    methods->tic();

    Tensor3f t_sharpness_smooth(t_sharpness->getInfoImgSeq(),"sharpness_smooth-STACK");

    //SMOOTH EACH LAYER
    Tensor3f t_mean15x15(15,"mean15x15-STACK"); //stack variable, gets destroyed anyway when function is left
    cudaConvolution(gridSize, blockSize, t_sharpness->getDevicePtr(), t_mean15x15.upload(), t_sharpness_smooth.getDevicePtrAllocated(), info.w, info.h, info.nrImgs,7); CUDA_CHECK; //7 = 15/2
    cudaDeviceSynchronize(); CUDA_CHECK;
    t_mean15x15.free();


    //FIND MAX SHARPNESS
    Tensor3f t_maxValuesSmooth(info.w,info.h,"maxValuesSmooth-STACK");  //not used in max...
    Tensor3f t_indicesMaxValuesSmooth(info.w,info.h,"indicesMaxValuesSmooth-STACK");

    cudaFindMax(gridSize,blockSize,t_sharpness_smooth.getDevicePtr(),t_maxValuesSmooth.getDevicePtrAllocated(),t_indicesMaxValuesSmooth.getDevicePtrAllocated(),info.w,info.h,info.nrImgs);
    cudaDeviceSynchronize(); CUDA_CHECK;
    t_maxValuesSmooth.free(); //never needed its values, just for kernel
    t_sharpness_smooth.free(); //not needed anymore from here on


    t_smoothDepthEstimate = new Tensor3f(info.w,info.h,"smoothDepthEstimate");

    Tensor3f t_mean21x21(21,"mean21x21-STACK");
    cudaConvolution(gridSize, blockSize, t_indicesMaxValuesSmooth.getDevicePtr(), t_mean21x21.upload(), t_smoothDepthEstimate->getDevicePtrAllocated(), info.w, info.h, 1, 10); CUDA_CHECK;
    cudaDeviceSynchronize(); CUDA_CHECK;
    t_mean21x21.free();
    t_indicesMaxValuesSmooth.free();

    methods->toc("smoothDepthEstimate");
    //out: smoothDepthEstimate
  }

  Mat DataPreparatorTensor3f::smoothDepthEstimate_ScaleIntoPolynomialRange(){
    //in: smoothDepthEstimate

    cout << "DataPreparatorTensor3f::smoothDepthEstimate_ScaleIntoPolynomialRange"<<endl;
    methods->tic();
  
    Mat mSmoothDepthEstimateScaled;

    if(scaleIntoPolynomialRange_GPU && false){ //TODO on GPU

    }else{
      t_smoothDepthEstimate->download();

      //scale into range of polynomial approximaton 
      Mat mTemp = t_smoothDepthEstimate->getMat() - 1.0f;
      mTemp /= static_cast<float>(info.nrImgs - 1);
      mTemp = mTemp - 0.5f;
      mSmoothDepthEstimateScaled = scale * mTemp;

      t_smoothDepthEstimate->setMat(mSmoothDepthEstimateScaled);
      t_smoothDepthEstimate->upload();
    }

    methods->toc("smoothDepthEstimate_ScaleIntoPolynomialRange");

    return mSmoothDepthEstimateScaled;

    //out: smoothDepthEstimate scaled to fit into polynomial range

  }
}