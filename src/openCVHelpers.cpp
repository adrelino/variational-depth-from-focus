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
#include <opencv2/contrib/contrib.hpp>

#include <iostream>

using namespace cv;
using namespace std;

namespace vdff {
  namespace openCVHelpers {
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

    void showDepthImage(const string &wndTitle, const Mat& img, int posX, int posY, bool doResize) {
      double min, max;
      minMaxIdx(img, &min, &max);

      Mat depthMap;
      float scale = 255.0f / (max - min);
      img.convertTo(depthMap, CV_8UC1, scale, -min*scale);

      Mat heatMap;
      applyColorMap(depthMap, heatMap, cv::COLORMAP_JET);

      if (doResize)
	resize(heatMap, heatMap, Size(), 0.5, 0.5);
  
      showImage(wndTitle, heatMap, posX, posY);
    }

    void createOptimallyPaddedImageForDCT(const Mat& img, Mat& paddedImg, 
					  int &paddingX, int &paddingY) {
      // pad init if it is not divisible by 2
      int maxVecSize = max(img.rows, img.cols);
      int optVecSize = getOptimalDFTSize((maxVecSize+1)/2)*2;

      paddingX = optVecSize - img.cols;
      paddingY = optVecSize - img.rows;

      int top, bottom, left, right;
      top = bottom = paddingY / 2;
      left = right = paddingX / 2;
      bottom += paddingY % 2 == 1;  
      right += paddingX % 2 == 1;

      if (paddingX == 0 && paddingY == 0) {
	paddedImg = img.clone();
      }
      else {
	copyMakeBorder(img, paddedImg, top, bottom, left, right, BORDER_CONSTANT, Scalar(0));
      }  
    }

    void imagesc(std::string title, cv::Mat mat, int x, int y) {
      double min,max;
      cv::minMaxLoc(mat,&min,&max);

      Mat scaled = mat;
      Mat meanCols;
      reduce(mat, meanCols, 0, CV_REDUCE_AVG );

      Mat mean;
      reduce(meanCols, mean, 1, CV_REDUCE_AVG);
    
      cout << "Max value: " << max << endl;
      cout << "Mean value: " << mean.at<float>(0) << endl;
      cout << "Min value: " << min << endl;
    
      if (std::abs(max) > 0.0000001f)
	scaled /= max;

      showImage(title, scaled, x,y);
    }

    void convert_layered_to_interleaved(float *aOut, const float *aIn, int w, int h, int nc)
    {
      if (nc==1) { memcpy(aOut, aIn, w*h*sizeof(float)); return; }
      size_t nOmega = (size_t)w*h;
      for (int y=0; y<h; y++)
	{
	  for (int x=0; x<w; x++)
	    {
	      for (int c=0; c<nc; c++)
		{
		  aOut[(nc-1-c) + nc*(x + (size_t)w*y)] = aIn[x + (size_t)w*y + nOmega*c];
		}
	    }
	}
    }

    void convert_layered_to_mat(cv::Mat &mOut, const float *aIn)
    {
      convert_layered_to_interleaved((float*)mOut.data, aIn, mOut.cols, mOut.rows, mOut.channels());
    }

    void convert_interleaved_to_layered(float *aOut, const float *aIn, int w, int h, int nc)
    {
      if (nc==1) { memcpy(aOut, aIn, w*h*sizeof(float)); return; }
      size_t nOmega = (size_t)w*h;
      for (int y=0; y<h; y++)
	{
	  for (int x=0; x<w; x++)
	    {
	      for (int c=0; c<nc; c++)
		{
		  aOut[x + (size_t)w*y + nOmega*c] = aIn[(nc-1-c) + nc*(x + (size_t)w*y)];
		}
	    }
	}
    }

    void convert_mat_to_layered(float *aOut, const cv::Mat &mIn)
    {
      convert_interleaved_to_layered(aOut, (float*)mIn.data, mIn.cols, mIn.rows, mIn.channels());
    }
  }
}
