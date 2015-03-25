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

#ifndef ID_H
#define ID_H
//compile for matlab with:   (in the src directory)
//nvcc -ptx common_kernels.cu -I ../include/ -DCUDA_MATLAB

//__host__ __device__ size_t id(const size_t x, const size_t y, const int w, const int h, const int channel=0);
//__host__ __device__ __forceinline__ size_t id(const size_t x, const size_t y, const int w, const int h, const int channel){
#ifdef CUDA_MATLAB
  //#pragma message ("compiling with MATLABS's col major order")
  #define id(x,y,w,h,channel) ((y) + (x)*(h) + (channel)*(h)*(w)) //note to self: pragma doesnt like default arguments/overloaded functions, since it is just a string replace basically
  #define id2(x,y,w,h) ((y) + (x)*(h))
#else
  //#pragma message ("compiling with OpenCV's row major order")
  #define id(x,y,w,h,channel) ((x) + (y)*(w) + (channel)*(w)*(h)) //parantheses needed if we pass negative values
  #define id2(x,y,w,h) ((x) + (y)*(w))
#endif

#endif //ID_H