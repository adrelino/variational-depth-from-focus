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

#ifndef DATA_PREPARATOR_H
#define DATA_PREPARATOR_H

#include <utils.cuh>
#include <opencv2/opencv.hpp>

namespace vdff {
  typedef enum{
    LAYERED,
    PIXELWISE
  } MemoryLayout;

  class DataPreparator {
  protected:
    const char *dirPath;
    MemoryLayout usedLayout;

    bool useMultipleStreams;
    Utils::InfoImgSeq info;
    float scale;

    float *l_sharpness;  
    float *l_maxValues;
    float *l_indicesMaxValues;

    float *d_coefDerivative;

    virtual void copyMLAPIntoMemory(float *d_MLAPEstimate, size_t index) = 0;
    virtual void copyMLAPIntoPageLockedMemory(float *d_MLAPEstimate, size_t index, cudaStream_t streamID=0) = 0;
  
    cv::Mat precomputePseudoInverse(const float *xi, size_t nrUnknowns, size_t polynomialDegree);
    virtual void copySharpnessImageToDevice(float *d_sharpness, size_t idx) = 0;
    virtual void copySmoothSharpnessImageToHost(float *d_smooth, size_t idx) = 0;

    void copySharpnessChunk(float *d_sharpness, size_t yOffset, size_t rowsToCopy, size_t nrBytesToCopy);

  void determineSharpnessFromAllImagesMultipleStreams(const std::vector<std::string> &imgFileNames, const cv::Mat& firstImage,
						      const Utils::Padding &padding,
                              size_t nrPixels, const int diffW, const int diffH, bool grayscale=false);
  
  void determineSharpnessFromAllImagesSingleStream(const std::vector<std::string> &imgFileNames, const cv::Mat& firstImage,
						   const Utils::Padding &padding,
                           size_t nrPixels, const int diffW, const int diffH, bool grayscale=false);

  public:
    DataPreparator(const char *dir, float minVal, float maxVal, MemoryLayout layout);
    virtual ~DataPreparator();

    void determineSharpnessFromAllImages(const cudaDeviceProp &deviceProperties, bool usePageLockedMemory, Utils::Padding &padding,
                     int skipNthPicture=1, bool grayscale=false);

    // split this up in separate classes
    virtual cv::Mat findMaxSharpnessValues() = 0;
    virtual void scaleSharpnessValues(float denomRegu) = 0;
    float *approximateContrastValuesWithPolynomial(size_t polynomialDegree);
    float *calcPolyApproximations(size_t polynomialDegree, const float denumRegu);
    cv::Mat smoothDepthEstimate(bool doChunked=false);
  
    float *getSharpnessValues();
    Utils::InfoImgSeq getInfoImgSeq();
  };
}
#endif
