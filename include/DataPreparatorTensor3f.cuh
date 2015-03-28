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

#ifndef DATA_PREPARATOR_TENSOR3f_H
#define DATA_PREPARATOR_TENSOR3f_H

#include <utils.cuh>
#include "Tensor3f.cuh"
#include <CPUTimer.h>

namespace vdff {
  class DataPreparatorTensor3f {
  protected:
    const char *dirPath;
  
    Utils::InfoImgSeq info;
    float scale;

    dim3 gridSize,blockSize; //gets initialized in call to determineSharpnessFromAllImages

    cv::Mat precomputePseudoInverse(const float *xi, size_t nrUnknowns, size_t polynomialDegree);

    CPUTimer *methods;
  public:
    DataPreparatorTensor3f(const char *dir, float minVal, float maxVal);
    virtual ~DataPreparatorTensor3f();

    cv::Mat determineSharpnessFromAllImages(int skipNthPicture=1, bool grayscale=false); //returns the last image in the sequence for seeing the colorchannel

    void findMaxSharpnessValues();

    void scaleSharpnessValues(float denomRegu);

    void approximateContrastValuesWithPolynomial(size_t polynomialDegree);
  
    void smoothDepthEstimate();

    cv::Mat smoothDepthEstimate_ScaleIntoPolynomialRange();


    Utils::InfoImgSeq getInfoImgSeq(){return info;};

    bool findMaxGPU;
    bool scaleSharpnessGPU;
    bool scaleIntoPolynomialRange_GPU;

    //used internally to pass data in between functions
    Tensor3f *t_imgSeq;
    Tensor3f *t_sharpness;
    Tensor3f *t_noisyDepthEstimate;
    Tensor3f *t_maxValues;


    //to pass on to ADMM
    Tensor3f *t_coefDerivative;
    Tensor3f *t_smoothDepthEstimate;
  };
}
#endif
