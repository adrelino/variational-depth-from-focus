#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <string>

namespace vdff {
  struct Parameters {
    std::string folderPath;

    // [minVal, maxVal] determines in which range we will do the
    // polynomial approximation later
    float minVal;
    float maxVal;
    size_t polynomialDegree;

    // value to decrease the importance of sharp edges
    float denomRegu; //0.8f no holes

    float dataFidelityParam;  //2.5f no holes
    float dataDescentStep;
    
    // nr iterations for ADMM algorithm
    size_t convIterations;
    size_t nrIterations;
    // lambda in ADMM procedure
    float lambda;
    int delay;

    bool useTensor3fClass;

    bool usePageLockedMemory;
    bool smoothGPU;

    int useNthPicture;
    std::string exportFilename;

  Parameters() : folderPath("../samples/sim"),
      minVal(-10.0f),
      maxVal(10.0f),
      polynomialDegree(6),
      denomRegu(0.3f),
      dataFidelityParam(6.0f),
      convIterations(0),
      nrIterations(400),
      lambda(1.0f),
      delay(1),
      useTensor3fClass(false),
      usePageLockedMemory(false),
      smoothGPU(true),
      useNthPicture(1),
      exportFilename("")
    {
      dataDescentStep = 8.0 / dataFidelityParam;
    }
  };
}
#endif
