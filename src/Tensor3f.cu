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

#include <Tensor3f.cuh>
#include <helper.h>
#include <openCVHelpers.h>

Tensor3f::Tensor3f(int w, int h, int nc, int nrImgs, std::string name)
{
	info.w = w;
	info.h = h;
	info.nc = nc;
	info.nrImgs = nrImgs;
	this->name=name;
	init();
}

Tensor3f::Tensor3f(int w, int h, std::string name)
{
	info.w = w;
	info.h = h;
	info.nc = 1;
	info.nrImgs = 1;
	this->name=name;
	init();
}

Tensor3f::Tensor3f(InfoImgSeq imgInfo, std::string name) 
#if __cplusplus > 199711L //http://stackoverflow.com/questions/10717502/is-there-a-preprocessor-directive-for-detecting-c11x-support
 : Tensor3f(imgInfo.w,imgInfo.h,imgInfo.nc, imgInfo.nrImgs, name) {}
#else
{
	info.w=imgInfo.w;
	info.h=imgInfo.h;
	info.nc=imgInfo.nc;
	info.nrImgs=imgInfo.nrImgs;
	this->name=name;
	init();
}
#endif

Tensor3f::Tensor3f(cv::Mat mat, std::string name)
{
	info.w = mat.cols;
	info.h = mat.rows;
	info.nc = mat.channels();
	info.nrImgs = 1;
	this->name=name;
	init();
	allocHost();
	convert_mat_to_layered(h_img,mat);
}

Tensor3f::Tensor3f(int size, std::string name)
{
	info.w = size;
	info.h = size;
	info.nc = 1;
	info.nrImgs = 1;
	this->name=name;
	init();
	allocHost();

	float val=1.0f/(size*size);
  	for (int i = 0; i < size*size; ++i) {
    	h_img[i] = val;
  	}
}

Tensor3f::~Tensor3f(){
	printf("Tensor3f[%s]::~Tensor3f:\n",/*test*/name.c_str());
	free();
}

void Tensor3f::init(){
	h_img_isAllocated=false;
	d_img_isAllocated=false;
	printf("Tensor3f[%s]::init\n",/*test*/name.c_str());
	nrPixels = info.w * info.h * info.nc * info.nrImgs;
	nrBytes = nrPixels * sizeof(float);
	printSize();
}

float* Tensor3f::upload(){
  printf("Tensor3f[%s]::upload\n",name.c_str());
  cudaDeviceSynchronize(); CUDA_CHECK;
  allocDevice();
  cudaMemcpy(d_img, h_img, nrBytes, cudaMemcpyHostToDevice); CUDA_CHECK;
  cudaDeviceSynchronize(); CUDA_CHECK;
  return getDevicePtr();
}

float* Tensor3f::download(){
  printf("Tensor3f[%s]::download\n",name.c_str());
  allocHost();
  cudaMemcpy(h_img, d_img, nrBytes,cudaMemcpyDeviceToHost); CUDA_CHECK;
  cudaDeviceSynchronize(); CUDA_CHECK;
  return getHostPtr();
}

void Tensor3f::allocHost(){
  if (!h_img_isAllocated) {
	  printf("Tensor3f[%s]::allocHost: %0.6f MB\n",/*test*/name.c_str(),nrBytes/1e6f);
	  h_img = new float[nrPixels];
	  h_img_isAllocated=true;
  }else{
  	  printf("Tensor3f[%s]::allocHost: was already allocated ------WARNING-------",/*test*/name.c_str());
  }
}

void Tensor3f::allocDevice(){
  if(!d_img_isAllocated){
	  printf("Tensor3f[%s]::allocDevice: %0.6f MB\n",/*test*/name.c_str(),nrBytes/1e6f);
	  cudaMalloc(&d_img, nrBytes); CUDA_CHECK;
	  d_img_isAllocated=true;
  }else{
  	  printf("Tensor3f[%s]::allocDevice: was already allocated ------WARNING-------",/*test*/name.c_str());
  }
}

void Tensor3f::freeHost(){
  if (h_img_isAllocated) {
    delete[] h_img;
    printf("Tensor3f[%s]::freeHost: %0.6f MB\n",/*test*/name.c_str(),nrBytes/1e6f);
    h_img_isAllocated = false;
  }
}

void Tensor3f::freeDevice(){
  if (d_img_isAllocated) {
    cudaFree(d_img);
    printf("Tensor3f[%s]::freeDevice: %0.6f MB\n",/*test*/name.c_str(),nrBytes/1e6f);
    d_img_isAllocated=false;
  }
}

void Tensor3f::printSize(){
  printf("Tensor3f[%s]::printSize [%d x %d x (%d * %d)] [w x h x (nc * nrImgs)] and size of %0.6f MB\n",/*test*/name.c_str(),info.w,info.h,info.nc,info.nrImgs,nrBytes/1e6f); //.6 gives us exactly the 6 digits of the bytes
}

void Tensor3f::setImageInSequence(int imgIndex, const cv::Mat slice){
	if(!h_img_isAllocated) allocHost();
	
	int nc=slice.channels();
	int type = slice.type();
	//printf("Tensor3f[%s]::setSlice: [%d x %d x %d] of type[%s / %d] to imgIndex[%d]\n",name.c_str(),slice.cols,slice.rows,nc,getImageType(type).c_str(),type,imgIndex);

	if(slice.rows != info.h || slice.cols != info.w || nc != info.nc || imgIndex >= info.nrImgs){
		printf("Tensor3f[%s]::setSlice: dimensions mismatch\n",/*test*/name.c_str());
		return;
	}

	//if(type != CV_32F || type != CV_32FC1 || type != CV_32FC3 ){ //So far only 1 and 3 channel float mats supported
	//	printf("Tensor3f[%s]::setSlice: type[%s]\n",/*test*/name.c_str(),getImageType(type).c_str());
		//return;
	//}
	//if(nc == 1 || nc == 3){
	//	printf("Tensor3f[%s]::setSlice: wrong #channels[%d]\n",/*test*/name.c_str(),nc);
	//	return;
	//}

	float* h_img_layerPtr = h_img + info.w * info.h * info.nc * imgIndex;

	convert_mat_to_layered(h_img_layerPtr, slice);
}

cv::Mat Tensor3f::getImageInSequence(int imgIndex){
  cv::Mat m_img = cv::Mat::zeros(info.h, info.w, CV_32FC1);
  if(imgIndex >= info.nrImgs){
  	printf("Tensor3f[%s]::getSlice: imgIndex[%d] >= nrImgs[%d]\n",/*test*/name.c_str(),imgIndex,info.nrImgs);
  	return m_img;
  }

  if(info.nc==1){
  	//m_img = cv::Mat::zeros(info.h,info.w,CV_32FC1); already done per default
  }else if(info.nc==3){
  	m_img = cv::Mat::zeros(info.h,info.w,CV_32FC3);
  }else{
  	printf("Tensor3f[%s]::getSlice: unsupported #channels[%d] to convert to OpenCV Mat\n",/*test*/name.c_str(),info.nc);
  }

  float* h_img_layerPtr = h_img + imgIndex * info.w * info.h * info.nc;
  convert_layered_to_mat(m_img, h_img_layerPtr);
  return m_img;
}