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

#include <openCVHelpers.h>
#include <iostream>

using namespace cv;
using namespace std;

void fillMatAscending(Mat& m) {
  CV_Assert(m.rows == 1 || m.cols == 1);

  for(size_t i=0; i < max(m.rows, m.cols); ++i) {
    m.at<float>(i) = static_cast<float>(i);
  }
}

float prod(const Mat& m) {
  CV_Assert(m.rows == 1 || m.cols == 1);
  
  float res = 1.0f;

  for(size_t i = 0; i < max(m.rows, m.cols); ++i) {
    res *= m.at<float>(i);
  }
  return res;
}

Mat reshapeColVector(const Mat& A, int rows, int cols) {
  Mat sz = Mat::zeros(1, 2, CV_32FC1);
  sz.at<float>(0) = static_cast<float>(rows);
  sz.at<float>(1) = static_cast<float>(cols);

  return reshapeColVector(A, sz);
}

Mat reshapeColVector(float *arr, int rows, int cols) {
  Mat res = Mat::zeros(rows, cols, CV_32FC1);

  int x = 0;
  int y = 0;
  for(int i = 0; i < rows*cols; ++i) {
    res.at<float>(y,x) = arr[i];
    ++y;
    if (y % rows == 0) {
      ++x;
      y = 0;
    }

  }

  return res;
}

Mat reshapeColVector(const Mat& A, const Mat& sz) {
  CV_Assert(A.channels() == 1 && min(A.rows, A.cols) == 1);
  CV_Assert(sz.channels() == 1 && sz.rows == 1);
  CV_Assert(static_cast<int>(prod(sz)) == A.total());

  int *dim = new int[sz.total()];
  for (int i = 0; i < sz.total(); ++i)
    dim[i] = static_cast<int>(sz.at<float>(i));

  Mat Res;
  Res.create(sz.total(), dim, CV_32FC1);

  int i = 0;
  for(int c = 0; c < Res.cols; ++c) {
    for (int y = 0; y < Res.rows; ++y) {
      Res(Rect(c, y, 1, 1)) = A.at<float>(i);
      ++i;
    }
  }

  delete[] dim;
  return Res;
}

//source: http://stackoverflow.com/questions/12335663/getting-enum-names-e-g-cv-32fc1-of-opencv-image-types
/*
std::string getImageType(int number) {
    // find type
    int imgTypeInt = number%8;
    std::string imgTypeString;

    switch (imgTypeInt)
    {
        case 0:
            imgTypeString = "8U";
            break;
        case 1:
            imgTypeString = "8S";
            break;
        case 2:
            imgTypeString = "16U";
            break;
        case 3:
            imgTypeString = "16S";
            break;
        case 4:
            imgTypeString = "32S";
            break;
        case 5:
            imgTypeString = "32F";
            break;
        case 6:
            imgTypeString = "64F";
            break;
        default:
            break;
    }

    // find channel
    int channel = (number/8) + 1;

    std::stringstream type;
    type<<"CV_"<<imgTypeString<<"C"<<channel;

    return type.str();
}
*/
// take number image type number (from cv::Mat.type()), get OpenCV's enum string.
std::string getImageType(int imgTypeInt)
{
    int numImgTypes = 35; // 7 base types, with five channel options each (none or C1, ..., C4)

    int enum_ints[] =       {CV_8U,  CV_8UC1,  CV_8UC2,  CV_8UC3,  CV_8UC4,
                             CV_8S,  CV_8SC1,  CV_8SC2,  CV_8SC3,  CV_8SC4,
                             CV_16U, CV_16UC1, CV_16UC2, CV_16UC3, CV_16UC4,
                             CV_16S, CV_16SC1, CV_16SC2, CV_16SC3, CV_16SC4,
                             CV_32S, CV_32SC1, CV_32SC2, CV_32SC3, CV_32SC4,
                             CV_32F, CV_32FC1, CV_32FC2, CV_32FC3, CV_32FC4,
                             CV_64F, CV_64FC1, CV_64FC2, CV_64FC3, CV_64FC4};

    string enum_strings[] = {"CV_8U",  "CV_8UC1",  "CV_8UC2",  "CV_8UC3",  "CV_8UC4",
                             "CV_8S",  "CV_8SC1",  "CV_8SC2",  "CV_8SC3",  "CV_8SC4",
                             "CV_16U", "CV_16UC1", "CV_16UC2", "CV_16UC3", "CV_16UC4",
                             "CV_16S", "CV_16SC1", "CV_16SC2", "CV_16SC3", "CV_16SC4",
                             "CV_32S", "CV_32SC1", "CV_32SC2", "CV_32SC3", "CV_32SC4",
                             "CV_32F", "CV_32FC1", "CV_32FC2", "CV_32FC3", "CV_32FC4",
                             "CV_64F", "CV_64FC1", "CV_64FC2", "CV_64FC3", "CV_64FC4"};

    for(int i=0; i<numImgTypes; i++)
    {
        if(imgTypeInt == enum_ints[i]) return enum_strings[i];
    }
    return "unknown image type";
}

void checkLoadedImage(const cv::Mat& m, const char *fileName) {
	if (!m.data) {
		cerr << "Could not find/open the file '" << fileName << "'." << endl;
		exit(EXIT_FAILURE);
	}
}

void showImage(const string &title, const cv::Mat &mat, int x, int y)
{
    const char *wTitle = title.c_str();
    cv::namedWindow(wTitle, CV_WINDOW_AUTOSIZE);
    cvMoveWindow(wTitle, x, y);
    cv::imshow(wTitle, mat);
}

#define MAX_RANGE(image) ((1 << 8*(image.elemSize1())) -1)

int convertToFloat(cv::Mat &image){
    int maxRange = MAX_RANGE(image);
    image.convertTo(image, CV_32F, 1.0f/maxRange);
//    cv::imshow("converted",image);
//    cv::waitKey();
    return maxRange;
}

void imgInfo(cv::Mat image){
    //double min,max;
    //cv::minMaxIdx(image,&min,&max);
    cout<<"\t type: "<<getImageType(image.type())<<"\t";
    cout<<"channels: "<<image.channels()<<"\t";
//    cout<<"depth: "<<image.depth()<<endl;
//    cout<<"elemSize: "<<image.elemSize()<<endl;
    cout<<"elemSize1: "<<image.elemSize1()<<" bytes \t";
    cout<<"maxRange: "<<MAX_RANGE(image)<<"\t";
    //cout<<"min: "<<min<<" \t max: "<<max<<endl;
//    cout<<"step: "<<image.step<<endl;
//    cout<<"step1: "<<image.step1()<<endl;
//    cout<<"total: "<<image.total()<<endl;
}

Mat imreadFloat(string filename){
    int flags = CV_LOAD_IMAGE_UNCHANGED;
//    if(openCVHelpers::color) flags = CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_COLOR;
//    if(openCVHelpers::grayscale) flags = CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_GRAYSCALE;

    cv::Mat original = imread(filename,flags);
    //cout<<endl;
    imgInfo(original);
    convertToFloat(original);
    //imgInfo(original);
    return original;
}

