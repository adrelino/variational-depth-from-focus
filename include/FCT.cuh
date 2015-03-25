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

#ifndef CUDA_FCT_H
#define CUDA_FCT_H

#include <helper.h>
#include <cufft.h>
#include <id.h>
#include <common_kernels.cuh>

class FCT {
private:
  //calculations vars
  cuFloatComplex *d_v;
  float *d_vreal;

  //precompiled vars
  cufftHandle planf, plani;
  cuFloatComplex *d_roots;
  size_t *d_v_index;

  //parameters
  dim3 blockSize;
  dim3 gridSize;
  int w,h;
  

public:
  FCT(int width, int height);
  ~FCT();

  void fct(float *d_input, float *d_output);
  void ifct(float *d_input, float *d_output);
};

#endif