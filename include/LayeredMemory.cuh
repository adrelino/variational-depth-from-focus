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

#ifndef LAYERED_MEMORY_CUH
#define LAYERED_MEMORY_CUH

#include <DataPreparator.cuh>

namespace vdff {
  class LayeredMemory : public DataPreparator {
  protected:
    virtual void copySharpnessImageToDevice(float *d_sharpness, size_t idx);
    virtual void copySmoothSharpnessImageToHost(float *d_smooth, size_t idx);
    virtual void copyMLAPIntoMemory(float *d_MLAPEstimate, size_t index);
    virtual void copyMLAPIntoPageLockedMemory(float *d_MLAPEstimate, size_t index, cudaStream_t streamID=0);  

  public:
    LayeredMemory(const char *dir, float minVal, float maxVal);
    ~LayeredMemory();

    virtual cv::Mat findMaxSharpnessValues();
    virtual void scaleSharpnessValues(float denomRegu);  
  };
}
#endif