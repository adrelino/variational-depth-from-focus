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

// class definition for the object which handles the inversion of (I - lambda * Laplacian)
#ifndef LAPLACE_INVERSION_H
#define LAPLACE_INVERSION_H

#include <opencv2/core/core.hpp>
#include <vector>
#include <FCT.cuh>

namespace vdff {
  class LaplaceInversion {
  private:
    // regularization parameter
    float lambda;

    // TODO(Dennis): remove the mat later - want to do everything on the device
    cv::Mat mKernel;
    cv::Mat denom;
  
    float *d_kernel;
    float *d_denom;

    FCT transform;

    std::vector<int> size;
    std::vector<int> hvec;

    // void initKernel();
    void initKernelGPU();

  public:
    LaplaceInversion();
    LaplaceInversion(const int* size, int sizeDim, const int* hvec, int hvecDim);
    LaplaceInversion(const std::vector<int> size, const std::vector<int> hvec);
    ~LaplaceInversion();
  
    float getLambda();
    void setLambda(float lambda);
    cv::Mat getKernel();
    cv::Mat solve(const cv::Mat& f);
    void solveGPU(float *d_f, float *d_res, size_t w, size_t h, size_t channels);
  };
}
#endif
