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

#ifndef HELPER_H
#define HELPER_H

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <cuda_runtime.h>
#include <ctime>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include <sstream>
#include <iostream>
#include "loading.h"

// opencv helpers
void convert_mat_to_layered(float *aOut, const cv::Mat &mIn);
void convert_layered_to_mat(cv::Mat &mOut, const float *aIn);

// measuring time
class Timer
{
    public:
	Timer() : tStart(0), running(false), sec(0.f)
	{
	}
	void start()
	{
		tStart = clock();
		running = true;
	}
	void end()
	{
		if (!running) { sec = 0; return; }
        cudaDeviceSynchronize();
		clock_t tEnd = clock();
		sec = (float)(tEnd - tStart) / CLOCKS_PER_SEC;
		running = false;
	}
	float get()
	{
		if (running) end();
		return sec;
	}
    private:
	clock_t tStart;
	bool running;
	float sec;
};




// cuda error checking
#define CUDA_CHECK cuda_check(__FILE__,__LINE__)
void cuda_check(std::string file, int line);


#endif  // HELPER_H
