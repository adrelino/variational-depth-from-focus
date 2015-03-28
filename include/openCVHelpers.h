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

// some openCV helper to make the code more MATLAB like ;-)
#ifndef OPENCV_HELPERS_H
#define OPENCV_HELPERS_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include <iostream>
#include <vector>

// for WINDOWS: has to be downloaded and put into ..\Visual Studio 2013\VC\include or similar
// check: http://www.softagalleria.net/download/dirent/
#include <dirent.h>
#include <sys/types.h>

#define MAX_RANGE(image) ((1 << 8*(image.elemSize1())) -1)

namespace vdff {
  namespace Utils {
    // parameter processing
    template<typename T> bool getParam(std::string param, T &var, int argc, char **argv, bool printParam=true)
      {
        const char *c_param = param.c_str();
        for(int i=argc-1; i>=1; i--)
	  {
	    if (argv[i][0]!='-') continue;
	    if (strcmp(argv[i]+1, c_param)==0)
	      {
		if (!(i+1<argc)) continue;
		std::stringstream ss;
		ss << argv[i+1];
		ss >> var;

		if (printParam) {
		  std::cout<<"PARAM[SET]: "<<param<<" : "<<var<<std::endl;
		}
	    
		return (bool)ss;
	      }
	  }
	if (printParam) {
	  std::cout<<"PARAM[DEF]: "<<param<<" : "<<var<<std::endl;
	}
        return false;
      }

    std::string getOSSeparator();
    std::vector<std::string> getAllImagesFromFolder(const char *dirname, int skipNthPicture=1);
  }


  namespace openCVHelpers {
    // fills Mat ascending with values like m = (0:1:m.cols)
    // assumes Mat is row/column vector
    void fillMatAscending(cv::Mat& m);

    // reproduce prod - but just for floats
    float prod(const cv::Mat& m);

    // reshapes column vector m into a matrix with dimension given
    // in size. ATTENTION: size has to be a row/column vector!
    cv::Mat reshapeColVector(const cv::Mat& A, int rows, int cols);
    cv::Mat reshapeColVector(float *arr, int rows, int cols);
    cv::Mat reshapeColVector(const cv::Mat& m, const cv::Mat& size);
    std::string getImageType(int number);

    void checkLoadedImage(const cv::Mat& m, const char *fileName);
    void showImage(const std::string &title, const cv::Mat &mat, int x, int y);

    void convert_mat_to_layered(float *aOut, const cv::Mat &mIn);
    void convert_layered_to_mat(cv::Mat &mOut, const float *aIn);
    void convert_interleaved_to_layered(float *aOut, const float *aIn, int w, int h, int nc);
    void convert_layered_to_interleaved(float *aOut, const float *aIn, int w, int h, int nc);

    void imagesc(std::string title, cv::Mat mat, int x, int y);
    void createOptimallyPaddedImageForDCT(const cv::Mat& img, cv::Mat& paddedImg, 
					  int &paddingX, int &paddingY);


    //passing these params ensures that same color implies same depth in different images
    //min = vdff::Parameters::minVal;
    //max = vdff::Parameters::maxVal;
    cv::Mat showDepthImage(const std::string &wndTitle, const cv::Mat& img, int posX, int posY, double min, double max,bool dResize=false);

    cv::Mat showDepthImage(const std::string &wndTitle, const cv::Mat& img, int posX, int posY, bool doResize=false);

    void exportImage(const cv::Mat &img, const std::string &fileName);

    int convertToFloat(cv::Mat &image);

    void imgInfo(cv::Mat image, bool full=false);
    //reads 1 or 3 channel image (with 8 or 16 bit depth) and converts it to 1 or 3 channel float (with 32 bit depth) correctly scaled form 0.0 -> 1.0f
    cv::Mat imreadFloat(std::string filename, bool grayscale=false);

    char waitKey2(int delay, bool hint=true);

  }
}
#endif
