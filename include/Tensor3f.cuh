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

#ifndef TENSOR3f
#define TENSOR3f

#include <utils.cuh>
#include <string>


/** 
 * Tensor of dimension 3 with all float entries. Abstracts away byte calculation. Interfaces to OpenCV's Mat as well as downlaod/upload functionality to get this onto the GPU
 * 
 * Trying to replicate http://docs.opencv.org/modules/gpu/doc/data_structures.html , but with 3 dimension
 * nice to have: http://docs.opencv.org/modules/gpu/doc/per_element_operations.html
 */
class Tensor3f {
public:
  Tensor3f(int w, int h, int nc=1, int nrImgs=1, std::string name="NoName"); //x,y,z
  Tensor3f(int w, int h, std::string name="NoName"); //x,y,z

  Tensor3f(InfoImgSeq imgSeqInfo, std::string name="NoName");
  //Tensor3f(ImgInfo imgInfo, std::string name="NoName");
  Tensor3f(cv::Mat mat, std::string name="NoName");
  Tensor3f(int size, std::string name="NoName"); //the mean kernel constructor, sets host memory

  virtual ~Tensor3f();

  float* upload();   //from cpu to gpu
  float* download(); //from gpu to cpu

  void allocDevice();
  void freeDevice();
  void allocHost();
  void freeHost();

  void alloc(){allocHost();allocDevice();};
  void free(){freeDevice();freeHost();};

  //Getters
  ImgInfo getImgInfo(){ImgInfo imgInfo = { info.w , info.h , info.nc*info.nrImgs }; return imgInfo;};  //TODO: rather check that imgSeqInfo.nrImgs==1?
  InfoImgSeq getInfoImgSeq(){return info;};
  float* getHostPtr(){return h_img;};
  float* getDevicePtr(){return d_img;};
  float* getHostPtrAllocated(){if(!h_img_isAllocated) allocHost();return h_img;};
  float* getDevicePtrAllocated(){if(!d_img_isAllocated) allocDevice();return d_img;};
  //cv::Mat's created on the fly
  cv::Mat getImageInSequence(int imgIndex);
  cv::Mat getMat(){return getImageInSequence(0);}; //gets the first image in sequence of nrImgs

  //Setters
  void setImageInSequence(int imgIndex, const cv::Mat slice); // up to 65535 in z dir
  void setMat(const cv::Mat slice){setImageInSequence(0,slice);};

  void printSize();

private:
  void init();

  std::string name;

  InfoImgSeq info; //dont store ImgInfo, that is just a InfoImqSeq with nrImgs=1;
  size_t nrPixels;
  size_t nrBytes;

  //CPU memory
  float *h_img; //(layered raw float array)
  bool h_img_isAllocated;

  //GPU memory
  float *d_img;
  bool d_img_isAllocated;

};

#endif