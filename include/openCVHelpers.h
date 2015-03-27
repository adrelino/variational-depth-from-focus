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

class openCVHelpers {
    public :
    static bool unchanged;
    static bool grayscale;
};

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

//void convertToFloat(cv::Mat& image);
void imgInfo(cv::Mat image);

//reads 1 or 3 channel image (with 8 or 16 bit depth) and converts it to 1 or 3 channel float (with 32 bit depth) correctly scaled form 0.0 -> 1.0f
cv::Mat imreadFloat(std::string filename);

#endif
