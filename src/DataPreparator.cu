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

#include <DataPreparator.cuh>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <openCVHelpers.h>

#include <cuda.h>
#include <cudaWrappers.h>
#include <iostream>
#include <new>

#include "openCVHelpers.h"

using namespace std;
using namespace cv;

namespace vdff {
  DataPreparator::DataPreparator(const char *dir, float min, float max, MemoryLayout layout) : dirPath(dir), usedLayout(layout),
											       l_sharpness(NULL), l_maxValues(NULL),
											       l_indicesMaxValues(NULL), d_coefDerivative(NULL),
											       scale(max - min), useMultipleStreams(false)
  {

  }

  DataPreparator::~DataPreparator() {
    if (d_coefDerivative != NULL) {
      cudaFree(d_coefDerivative);
      d_coefDerivative = NULL;
    }

    if (l_maxValues != NULL) {
      delete[] l_maxValues;
      l_maxValues = NULL;
    }

    if (l_indicesMaxValues != NULL) {
      delete[] l_indicesMaxValues;
      l_indicesMaxValues = NULL;    
    }
  
    if (l_sharpness != NULL) {
      if (useMultipleStreams)
	cudaFreeHost(l_sharpness);
      else
	delete[] l_sharpness;
      l_sharpness = NULL;
    }
  }

  void DataPreparator::determineSharpnessFromAllImagesSingleStream(const vector<string> &imgFileNames, const Mat& firstImage,
								   const Utils::Padding &padding,
								   size_t nrPixels, const int diffW, const int diffH,bool grayscale) {
    cout << "Executing with a single stream" << endl;
    Mat curImg = firstImage.clone();
    int numberChannelsImage = curImg.channels();

    size_t nrBytes = nrPixels * sizeof(float);
    
    float *l_img = new float[nrPixels];
    if (l_img == NULL) {
      throw std::runtime_error("could not allocate memory for l_img");
    }
    
    float *d_img;
    cudaMalloc(&d_img, nrBytes); CUDA_CHECK;
  
    // create big array on gpu memory for all sharpness values
    // sharpness images have only one channel
    size_t sharpnessPixels = static_cast<size_t>(info.w*info.h*info.nc);
    size_t sharpnessBytes = sharpnessPixels*sizeof(float);

    // ATTENTION: we do no memset, since we are sure that we are overwriting everything,
    // so there is no need to set it to zero --> saving some time.
    float *d_sharpnessCurImg;
    cudaMalloc(&d_sharpnessCurImg, sharpnessBytes); CUDA_CHECK;

    // use here host allocated memory
    try {
      l_sharpness = new float[sharpnessPixels * info.nrImgs];
    }
    catch (bad_alloc &badAllocEx) {
      cerr << "Can not allocate enough memory to store all sharpness values: " << badAllocEx.what() << endl;
      cerr << "Aborting..." << endl;
      exit(1);
    }

    dim3 blockSize(32, 8, 1);
    dim3 gridSize((info.w + blockSize.x - 1) / blockSize.x, (info.h + blockSize.y - 1) / blockSize.y, 1);

    string imgFile = "";
  
    for(size_t i = 0; i < info.nrImgs; ++i) {
      cout << "\r" << flush;
      cout << "Determining sharpness from picture " << (i+1) << " from " << info.nrImgs;

      if (i != 0) {
	imgFile = imgFileNames[i];
	curImg = openCVHelpers::imreadFloat(imgFileNames[i],grayscale);
      
	// pad the image if we need to, to guarantee that DCT is possible
	if (diffW != 0 || diffH != 0) {
	  copyMakeBorder(curImg, curImg, padding.top, padding.bottom, padding.left, padding.right, BORDER_REPLICATE);
	}
      }
      openCVHelpers::convert_mat_to_layered(l_img, curImg);

      cudaMemcpy(d_img, l_img, nrBytes, cudaMemcpyHostToDevice); CUDA_CHECK;
      cudaComputeMLAP(gridSize, blockSize, d_img, d_sharpnessCurImg, info.w, info.h, numberChannelsImage); CUDA_CHECK;
      copyMLAPIntoMemory(d_sharpnessCurImg, i);
    }
    cout << endl;

    cudaFree(d_img);
    cudaFree(d_sharpnessCurImg);
  
    delete[] l_img;
  }

  void DataPreparator::determineSharpnessFromAllImagesMultipleStreams(const vector<string> &imgFileNames, const Mat& firstImage,
								      const Utils::Padding& padding,
								      size_t nrPixels, const int diffW, const int diffH, bool grayscale) {
    cout << "Executing with 2 streams" << endl;
    Mat curImg = firstImage.clone();
    int numberChannelsImage = curImg.channels();

    size_t nrBytes = nrPixels * sizeof(float);

    // float *l_img = new float[nrPixels];
    float *l_img0, *l_img1;
    cudaHostAlloc((void**)&l_img0, nrPixels*sizeof(float), cudaHostAllocDefault); CUDA_CHECK;
    cudaHostAlloc((void**)&l_img1, nrPixels*sizeof(float), cudaHostAllocDefault); CUDA_CHECK;

    float *d_img0, *d_img1;
    cudaMalloc(&d_img0, nrBytes); CUDA_CHECK;
    cudaMalloc(&d_img1, nrBytes); CUDA_CHECK;

    // create big array on gpu memory for all sharpness values
    // sharpness images have only one channel
    size_t sharpnessPixels = static_cast<size_t>(info.w*info.h*info.nc);
    size_t sharpnessBytes = sharpnessPixels*sizeof(float);

    // ATTENTION: we do no memset, since we are sure that we are overwriting everything,
    // so there is no need to set it to zero --> saving some time.
    float *d_sharpnessCurImg0, *d_sharpnessCurImg1;
    cudaMalloc(&d_sharpnessCurImg0, sharpnessBytes); CUDA_CHECK;
    cudaMalloc(&d_sharpnessCurImg1, sharpnessBytes); CUDA_CHECK;

    cudaStream_t stream0;
    cudaStream_t stream1;
    cudaStreamCreate(&stream0); CUDA_CHECK;
    cudaStreamCreate(&stream1); CUDA_CHECK;

    // use here host allocated memory
    cudaHostAlloc((void**)&l_sharpness, sharpnessPixels * info.nrImgs * sizeof(float), cudaHostAllocDefault); CUDA_CHECK;

    dim3 blockSize(32, 8, 1);
    dim3 gridSize((info.w + blockSize.x - 1) / blockSize.x, (info.h + blockSize.y - 1) / blockSize.y, 1);

    string imgFile = "";

    for(size_t i = 0; i < info.nrImgs; i+=2) {
      cout << "\r" << flush;
      cout << "Determining sharpness from picture " << (i+1) << " from " << info.nrImgs;

      if (i != 0) {
	imgFile = imgFileNames[i];
	curImg = openCVHelpers::imreadFloat(imgFileNames[i],grayscale);
      }

      // check if we got the same size, before the (possible) padding!
      assert(firstImage.cols == curImg.cols && firstImage.rows == curImg.rows && firstImage.channels() == curImg.channels());

      // pad the image if we need to, to guarantee that DCT is possible
      if (diffW != 0 || diffH != 0)
	copyMakeBorder(curImg, curImg, padding.top, padding.bottom, padding.left, padding.right, BORDER_REPLICATE);

      openCVHelpers::convert_mat_to_layered(l_img0, curImg);

      size_t iPlusOne = i+1;
      bool isIPlusOneValid = iPlusOne < info.nrImgs;

      if (isIPlusOneValid) {
	imgFile = imgFileNames[iPlusOne];
	curImg = openCVHelpers::imreadFloat(imgFileNames[iPlusOne],grayscale);

	// check if we got the same size, before the (possible) padding!
	assert(firstImage.cols == curImg.cols && firstImage.rows == curImg.rows && firstImage.channels() == curImg.channels());

	// pad the image if we need to, to guarantee that DCT is possible
	if (diffW != 0 || diffH != 0)
	  copyMakeBorder(curImg, curImg, padding.top, padding.bottom, padding.left, padding.right, BORDER_REPLICATE);

	openCVHelpers::convert_mat_to_layered(l_img1, curImg);
      }

      cudaMemcpyAsync(d_img0, l_img0, nrBytes, cudaMemcpyHostToDevice, stream0); CUDA_CHECK;
      if (isIPlusOneValid)
	cudaMemcpyAsync(d_img1, l_img1, nrBytes, cudaMemcpyHostToDevice, stream1); CUDA_CHECK;

      cudaComputeMLAP(gridSize, blockSize, d_img0, d_sharpnessCurImg0, info.w, info.h, numberChannelsImage, 1, stream0); CUDA_CHECK;
      if (isIPlusOneValid)
	cudaComputeMLAP(gridSize, blockSize, d_img1, d_sharpnessCurImg1, info.w, info.h, numberChannelsImage, 1, stream1); CUDA_CHECK;

      copyMLAPIntoPageLockedMemory(d_sharpnessCurImg0, i, stream0);

      if (isIPlusOneValid)
	copyMLAPIntoPageLockedMemory(d_sharpnessCurImg1, iPlusOne, stream1);
    }
    cudaStreamSynchronize(stream0); CUDA_CHECK;
    cudaStreamSynchronize(stream1); CUDA_CHECK;
    cout << endl;

    cudaFree(d_img0);
    cudaFree(d_img1);

    cudaFree(d_sharpnessCurImg0);
    cudaFree(d_sharpnessCurImg1);

    cudaFreeHost(l_img0);
    cudaFreeHost(l_img1);

    cudaStreamDestroy(stream0);
    cudaStreamDestroy(stream1);
  }

  void DataPreparator::determineSharpnessFromAllImages(const cudaDeviceProp &deviceProperties, bool usePageLockedMemory,  Utils::Padding &imgPadding, int useNthPicture, bool grayscale) {
    vector<string> imgFileNames = Utils::getAllImagesFromFolder(dirPath, useNthPicture);

    size_t nrImgs = imgFileNames.size();
    string imgFile = imgFileNames[0];

    cout << "loading img: " << imgFileNames[0] << endl;
    Mat curImg = openCVHelpers::imreadFloat(imgFileNames[0],grayscale);
    cout << "\tdone" << endl;
    
    int w = curImg.cols;
    int h = curImg.rows;
    int nc = curImg.channels();

    // determine here if we need a padding
    // since we are assuming all images have same size!
    int optW = getOptimalDFTSize((w+1)/2)*2;
    int optH = getOptimalDFTSize((h+1)/2)*2;

    int diffW = optW - w;
    int diffH = optH - h;

    imgPadding.top = imgPadding.bottom = diffH / 2;
    imgPadding.left = imgPadding.right = diffW / 2;
    imgPadding.bottom += diffH % 2 == 1;
    imgPadding.right += diffW % 2 == 1;

    if (diffW != 0 || diffH != 0) {
      copyMakeBorder(curImg, curImg, imgPadding.top, imgPadding.bottom, imgPadding.left, imgPadding.right, BORDER_REPLICATE);
    }
  
    // fill info struct
    info.w = curImg.cols;
    info.h = curImg.rows;
    info.nrImgs = nrImgs;
    // ATTENTION: we have only one channel in the sharpness image!
    info.nc = 1;

    size_t nrPixels = static_cast<size_t>(curImg.cols*curImg.rows*curImg.channels());

    // check if the graphics device can handle overlaps --> no: only use one stream (default), else: do it with multiple streams
    // (at the moment 2)
    if (!deviceProperties.deviceOverlap || !usePageLockedMemory) {
      cout << "determine sharpness - single stream" << endl;
      useMultipleStreams = false;
      determineSharpnessFromAllImagesSingleStream(imgFileNames, curImg,
						  imgPadding,
						  nrPixels, diffW, diffH,grayscale);
    } else {
      useMultipleStreams = true;
      cout << "determine sharpness - multi stream" << endl;
      determineSharpnessFromAllImagesMultipleStreams(imgFileNames, curImg,
						     imgPadding,
						     nrPixels, diffW, diffH,grayscale);
    }
  }

  Mat DataPreparator::precomputePseudoInverse(const float *xi, size_t nrUnknowns, size_t polynomialDegree) {
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

  void DataPreparator::copySharpnessChunk(float *d_sharpness, size_t yOffset, size_t heightOnDevice, size_t bytesToCopy) {
    size_t bytesPerLayer = bytesToCopy / info.nrImgs;

    for(size_t n = 0; n < info.nrImgs; ++n) {
      // index into d_sharpness
      size_t startIdx = n*heightOnDevice*info.w;
      size_t copyIdx = yOffset*info.w + n*info.w*info.h;

      cudaMemcpy(&d_sharpness[startIdx], &l_sharpness[copyIdx], bytesPerLayer, cudaMemcpyHostToDevice); CUDA_CHECK;
    }
  }

  float *DataPreparator::calcPolyApproximations(size_t degree, const float denomRegu) {
    dim3 blockSize(32, 8, 1);
    dim3 gridSize((info.w + blockSize.x -1) / blockSize.x, (info.h + blockSize.y -1) / blockSize.y, 1);
  
    // fixed memory which we need on the GPU
    // create xi's
    float *xi = new float[info.nrImgs];
    for(int i = 0; i < info.nrImgs; ++i) {
      xi[i] = scale * ((i / static_cast<float>(info.nrImgs-1)) - 0.5f);
    }

    // compute pseudo-inverse
    Mat mpInv = precomputePseudoInverse(xi, info.nrImgs, degree);
    delete[] xi;
  
    float *l_pInv = new float[mpInv.cols*mpInv.rows];
    openCVHelpers::convert_mat_to_layered(l_pInv, mpInv);
  
    float *d_pInv;
    cudaMalloc(&d_pInv, mpInv.cols * mpInv.rows * sizeof(float)); CUDA_CHECK;
    cudaMemcpy(d_pInv, l_pInv, mpInv.cols * mpInv.rows * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
    delete[] l_pInv;
  
    float *d_negCoef;
    cudaMalloc(&d_negCoef, info.w * info.h * (degree+1) * sizeof(float)); CUDA_CHECK;
    cudaMemset(d_negCoef, 0, info.w * info.h * (degree+1) * sizeof(float)); CUDA_CHECK;

    cudaMalloc(&d_coefDerivative, info.w * info.h * degree * sizeof(float)); CUDA_CHECK;
    cudaMemset(d_coefDerivative, 0, info.w * info.h * degree * sizeof(float)); CUDA_CHECK;

    // check how much memory we got available at the moment
    size_t freeMemory, totalMemory;
    Utils::getAvailableGlobalMemory(&freeMemory, &totalMemory);

    size_t bytesPerPixel = info.nrImgs * sizeof(float);
    size_t bytesPerRow = info.w * bytesPerPixel;

    size_t rowsPerKernel = floorf(static_cast<float>(freeMemory) / bytesPerRow);

    // compute space for needed arrays
    size_t sizeNeededForMaxValues = info.w * rowsPerKernel * sizeof(float);
    size_t sizeNeededForMaxIndices = info.w * rowsPerKernel * sizeof(int);
    size_t actualFreeMemory = freeMemory - sizeNeededForMaxValues - sizeNeededForMaxIndices;

    size_t actualRowsPerKernel = floorf(static_cast<float>(actualFreeMemory) / bytesPerRow) - 10;
    size_t nrChunks = ceilf(static_cast<float>(info.h) / actualRowsPerKernel);

    float *d_sharpness;
    float *d_maxValues;
    float *d_maxIndices;

    if (nrChunks == 1) {
      // allocate memory for sharpness values
      size_t totalBytes = bytesPerRow*info.h;
      cudaMalloc(&d_sharpness, totalBytes); CUDA_CHECK;
      cudaMemcpy(d_sharpness, l_sharpness, totalBytes, cudaMemcpyHostToDevice); CUDA_CHECK;
    
      // reserve space for max-kernel
      cudaMalloc(&d_maxValues, info.w*info.h*sizeof(float)); CUDA_CHECK;
      cudaMemset(d_maxValues, 0, info.w*info.h*sizeof(float)); CUDA_CHECK;

      cudaMalloc(&d_maxIndices, info.w*info.h*sizeof(float)); CUDA_CHECK;
      cudaMemset(d_maxIndices, 0, info.w*info.h*sizeof(float)); CUDA_CHECK;

      // run kernel to determine max and maxIndices
      findMax<<<gridSize, blockSize>>>(d_sharpness, d_maxValues, d_maxIndices, info.w, info.h, info.nrImgs); CUDA_CHECK;
      cudaDeviceSynchronize(); CUDA_CHECK;
    
      // now scale them accordingly
      scaleSharpnessValuesGPU<<<gridSize, blockSize>>>(d_maxValues, d_sharpness, info.w, info.h, info.nrImgs, denomRegu); CUDA_CHECK;
      cudaDeviceSynchronize(); CUDA_CHECK;    

      // do now the fitting!
      cudaPolyfit(gridSize, blockSize, d_pInv, d_sharpness, d_negCoef,
		  info.w, info.h, info.nrImgs, degree+1, usedLayout);
      cudaDeviceSynchronize(); CUDA_CHECK;
    } else {
      cout << "Using " << nrChunks << " chunks to approximate contrast values" << endl;

      size_t bytesToProcess = bytesPerRow * info.h;
      size_t processedRows = 0;
      // bytes to copy for every chunk
      size_t heightOnDevice = (actualRowsPerKernel > info.h) ? info.h : actualRowsPerKernel;
      size_t bytesToCopy = heightOnDevice * bytesPerRow;

      dim3 blockSizeChunked(32, 8, 1);
      dim3 gridSizeChunked((info.w + blockSizeChunked.x -1) / blockSizeChunked.x,
			   (heightOnDevice + blockSizeChunked.y -1) / blockSizeChunked.y, 1);

      cudaMalloc(&d_sharpness, info.w*heightOnDevice*info.nrImgs*sizeof(float)); CUDA_CHECK;
      cudaMemset(d_sharpness, 0, info.w*heightOnDevice*info.nrImgs*sizeof(float)); CUDA_CHECK;

      // reserve space for max-kernel
      cudaMalloc(&d_maxValues, info.w*heightOnDevice*sizeof(float)); CUDA_CHECK;
      cudaMemset(d_maxValues, 0, info.w*heightOnDevice*sizeof(float)); CUDA_CHECK;

      cudaMalloc(&d_maxIndices, info.w*heightOnDevice*sizeof(int)); CUDA_CHECK;
      cudaMemset(d_maxIndices, 0, info.w*heightOnDevice*sizeof(int)); CUDA_CHECK;

      for(size_t chunk = 0; chunk < nrChunks; ++chunk) {
	cout << "\r" << flush;
	cout << "Computing chunk " << (chunk+1);
	copySharpnessChunk(d_sharpness, processedRows, heightOnDevice, bytesToCopy);
	cudaDeviceSynchronize(); CUDA_CHECK;

	findMax<<<gridSize, blockSize>>>(d_sharpness, d_maxValues, d_maxIndices, info.w, heightOnDevice, info.nrImgs); CUDA_CHECK;
	cudaDeviceSynchronize(); CUDA_CHECK;
    
	// now scale them accordingly
	scaleSharpnessValuesGPU<<<gridSize, blockSize>>>(d_maxValues, d_sharpness, info.w, heightOnDevice, info.nrImgs, denomRegu); CUDA_CHECK;
	cudaDeviceSynchronize(); CUDA_CHECK;
    
	matrixPolyfitChunked<<<gridSizeChunked, blockSizeChunked>>>(d_pInv, d_sharpness, d_negCoef,
								    info.w, heightOnDevice, info.nrImgs,
								    info.w, info.h, degree+1, processedRows); CUDA_CHECK;    
	cudaDeviceSynchronize(); CUDA_CHECK;

	processedRows += heightOnDevice;

	// determine how much bytes we have to copy in the next chunk
	bytesToProcess -= bytesToCopy;    
	if (bytesToProcess < bytesToCopy) {
	  bytesToCopy = bytesToProcess;

	  size_t remainingHeight = bytesToCopy / bytesPerRow;
	  heightOnDevice = remainingHeight;
	  cudaFree(d_sharpness);
	  cudaMalloc(&d_sharpness, info.w*heightOnDevice*info.nrImgs*sizeof(float)); CUDA_CHECK;

	  bytesToProcess = 0;
	}
      }
      cout << endl;
      assert(processedRows == info.h);
    }

    cudaPolyder(gridSize, blockSize, d_negCoef, d_coefDerivative, info.w, info.h, degree+1, usedLayout); CUDA_CHECK;
    cudaDeviceSynchronize(); CUDA_CHECK;
  
    // negate the derivatives
    cudaMultiplyArrayWithScalar(gridSize, blockSize, d_coefDerivative, -1.0f, d_coefDerivative, info.w, info.h, degree); CUDA_CHECK;
    cudaDeviceSynchronize(); CUDA_CHECK;

    cudaFree(d_sharpness);  
    cudaFree(d_pInv);
    cudaFree(d_negCoef);
  
    cudaFree(d_maxValues);
    cudaFree(d_maxIndices);

    return d_coefDerivative;
  }

  Mat DataPreparator::smoothDepthEstimate(bool doChunked) {
    float *d_sharpness;
    float *d_smooth;
    float *d_mean15x15, *d_mean21x21;

    size_t nrPixels = static_cast<size_t>(info.w*info.h);
    size_t nrBytes = nrPixels * sizeof(float);

    float *l_smoothDepthEstimate = new float[nrPixels];

    float *mean15x15 = new float[15*15];
    for (int i = 0; i < 15*15; ++i) {
      mean15x15[i] = 1.0f/(15.0f*15.0f);
    }

    float *mean21x21 = new float[21*21];
    for (int i = 0; i < 21*21; ++i) {
      mean21x21[i] = 1.0f/(21.0f*21.0f);
    }  

    cudaMalloc(&d_sharpness, nrBytes); CUDA_CHECK;
    cudaMemset(d_sharpness, 0, nrBytes); CUDA_CHECK;

    cudaMalloc(&d_smooth, nrBytes); CUDA_CHECK;
    cudaMemset(d_smooth, 0, nrBytes); CUDA_CHECK;

    cudaMalloc(&d_mean15x15, 15*15*sizeof(float)); CUDA_CHECK;
    cudaMemcpy(d_mean15x15, mean15x15, 15*15*sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
  
    cudaMalloc(&d_mean21x21, 21*21*sizeof(float)); CUDA_CHECK;
    cudaMemcpy(d_mean21x21, mean21x21, 21*21*sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;  

    dim3 blockSize(32, 8, 1);
    dim3 gridSize((info.w + blockSize.x -1) / blockSize.x, (info.h + blockSize.y -1) / blockSize.y, 1);  

    cout << "Convolving each image with 15x15 mean filter..." << endl;
    for(int i = 0; i < info.nrImgs; ++i) {
      copySharpnessImageToDevice(d_sharpness, i);
      cudaConvolution(gridSize, blockSize, d_sharpness, d_mean15x15, d_smooth, info.w, info.h, 1,
		      7); CUDA_CHECK;
      cudaDeviceSynchronize(); CUDA_CHECK;
      copySmoothSharpnessImageToHost(d_smooth, i);
      cout << "\r" << flush;
      cout << "\tdone with image " << (i+1);
    }
    cout << endl;
  
    cudaFree(d_mean15x15);
    delete[] mean15x15;

    cout << "Convolving smoothed images with 21x21 mean filter..." << endl;  
    if (doChunked) {
      // allocate memory for maxIndices
      float *d_maxIndices;
      cudaMalloc(&d_maxIndices, info.w*info.h*sizeof(float)); CUDA_CHECK;
      cudaMemset(d_maxIndices, 0, info.w*info.h*sizeof(float)); CUDA_CHECK;

      size_t freeMemory, totalMemory;
      Utils::getAvailableGlobalMemory(&freeMemory, &totalMemory);

      size_t bytesPerPixel = info.nrImgs * sizeof(float);
      size_t bytesPerRow = info.w * bytesPerPixel;

      size_t rowsPerKernel = floorf(static_cast<float>(freeMemory) / bytesPerRow) - 10;
      // compute space for needed arrays
      size_t nrChunks = ceilf(static_cast<float>(info.h) / rowsPerKernel);

      if (nrChunks == 1) {
	// allocate memory for sharpness values
	size_t totalBytes = bytesPerRow*info.h;
	cudaMalloc(&d_sharpness, totalBytes); CUDA_CHECK;
	cudaMemcpy(d_sharpness, l_sharpness, totalBytes, cudaMemcpyHostToDevice); CUDA_CHECK;
    
	// run kernel to determine max and maxIndices
	findMaxIndices<<<gridSize, blockSize>>>(d_sharpness, d_maxIndices, info.w, info.h, info.nrImgs,
						info.w, info.h, 0); CUDA_CHECK;
	cudaDeviceSynchronize(); CUDA_CHECK;
      } else {
	cout << "Using " << nrChunks << " chunks to create initial smooth depth estimate " << endl;
	size_t bytesToProcess = bytesPerRow * info.h;
	size_t processedRows = 0;
	// bytes to copy for every chunk
	size_t heightOnDevice = rowsPerKernel;
	size_t bytesToCopy = heightOnDevice * bytesPerRow;

	dim3 blockSizeChunked(32, 8, 1);
	dim3 gridSizeChunked((info.w + blockSizeChunked.x -1) / blockSizeChunked.x,
			     (heightOnDevice + blockSizeChunked.y -1) / blockSizeChunked.y, 1);

	cudaMalloc(&d_sharpness, info.w*heightOnDevice*info.nrImgs*sizeof(float)); CUDA_CHECK;
	cudaMemset(d_sharpness, 0, info.w*heightOnDevice*info.nrImgs*sizeof(float)); CUDA_CHECK;

	for(size_t chunk = 0; chunk < nrChunks; ++chunk) {
	  cout << "\r" << flush;
	  cout << "Computing chunk " << (chunk+1);
	  copySharpnessChunk(d_sharpness, processedRows, heightOnDevice, bytesToCopy);
	  cudaDeviceSynchronize(); CUDA_CHECK;

	  findMaxIndices<<<gridSize, blockSize>>>(d_sharpness, d_maxIndices, info.w, heightOnDevice, info.nrImgs,
						  info.w, info.h, processedRows); CUDA_CHECK;
	  cudaDeviceSynchronize(); CUDA_CHECK;

	  processedRows += heightOnDevice;

	  // determine how much bytes we have to copy in the next chunk
	  bytesToProcess -= bytesToCopy;    
	  if (bytesToProcess < bytesToCopy) {
	    bytesToCopy = bytesToProcess;

	    size_t remainingHeight = bytesToCopy / bytesPerRow;
	    heightOnDevice = remainingHeight;
	    cudaFree(d_sharpness);
	    cudaMalloc(&d_sharpness, info.w*heightOnDevice*info.nrImgs*sizeof(float)); CUDA_CHECK;

	    bytesToProcess = 0;
	  }
	}
	cout << endl;      
	assert(processedRows == info.h);
      }

      cudaConvolution(gridSize, blockSize, d_maxIndices, d_mean21x21, d_smooth, info.w, info.h, 1, 10); CUDA_CHECK;
      cudaMemcpy(l_smoothDepthEstimate, d_smooth, nrBytes, cudaMemcpyDeviceToHost); CUDA_CHECK;
      cudaFree(d_maxIndices);  
    } else {
      findMaxSharpnessValues();
      cudaMemcpy(d_sharpness, l_indicesMaxValues, info.w * info.h * sizeof(float), cudaMemcpyHostToDevice);
    
      cudaConvolution(gridSize, blockSize, d_sharpness, d_mean21x21, d_smooth, info.w, info.h, 1, 10); CUDA_CHECK;
      cudaMemcpy(l_smoothDepthEstimate, d_smooth, nrBytes, cudaMemcpyDeviceToHost); CUDA_CHECK;
      cudaDeviceSynchronize(); CUDA_CHECK;
    }
  
    cudaFree(d_sharpness);
    cudaFree(d_smooth);
    cudaFree(d_mean21x21);
  
    delete[] mean21x21;

    delete[] l_indicesMaxValues;
    l_indicesMaxValues = NULL;
  
    delete[] l_maxValues;
    l_maxValues = NULL;

    Mat mSmoothDepthEstimate = Mat::zeros(info.h, info.w, CV_32FC1);
    openCVHelpers::convert_layered_to_mat(mSmoothDepthEstimate, l_smoothDepthEstimate);
  
    Mat mTemp = mSmoothDepthEstimate - 1.0f;
    mTemp /= static_cast<float>(info.nrImgs - 1);
    mTemp = mTemp - 0.5f;
    Mat mSmoothDepthEstimateScaled = scale * mTemp;
  
    delete[] l_smoothDepthEstimate;

    return mSmoothDepthEstimateScaled;
  }

  float *DataPreparator::getSharpnessValues() {
    return l_sharpness;
  }

  Utils::InfoImgSeq DataPreparator::getInfoImgSeq() {
    return info;
  }
}

