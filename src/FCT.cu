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

#include <FCT.cuh>

namespace vdff {
  //initializes the object, allocates device memory
  FCT::FCT(int width, int height){
    w=width;
    h=height;

    blockSize=dim3(32,8,1);
    gridSize=dim3((w + blockSize.x - 1) / blockSize.x, (h + blockSize.y - 1) / blockSize.y, 1);

    //allocate cufft plans
    cufftPlan2d(&planf, h, w, CUFFT_R2C); CUDA_CHECK;
    cufftPlan2d(&plani, h, w, CUFFT_C2R); CUDA_CHECK;

    //precompute unit roots
    cudaMalloc((void**)&d_roots, 2*(w+h)*sizeof(cuFloatComplex)); CUDA_CHECK;
    cuFCTinitRoots<<<dim3((2*(w+h) + blockSize.x - 1) / blockSize.x, 1, 1),blockSize>>>(d_roots, w, h);

    //precompute indices for resorting
    cudaMalloc((void**)&d_v_index, w*h*sizeof(size_t)); CUDA_CHECK;
    cuFCTinitIndex<<<gridSize,blockSize>>>(d_v_index, w, h);

    //allocate calculation memory
    cudaMalloc((void**)&d_v, (w / 2 + 1)*h*sizeof(cuFloatComplex)); CUDA_CHECK;
    cudaMalloc((void**)&d_vreal, w*h*sizeof(float)); CUDA_CHECK;
  }

  FCT::~FCT(){
    cufftDestroy(planf);
    cufftDestroy(plani);
    cudaFree(d_roots);
    cudaFree(d_v);
    cudaFree(d_vreal);
    cudaFree(d_v_index);
  }


  ////
  //computes the Fast Discrete Cosine Transformation
  //
  //d_input, d_output = layered quadratic 2d w x h gpu array
  //requires gpu allocated memory + instance of FCT
  void FCT::fct(float *d_input, float *d_output) {
    //NxN 2d FCT using NxN FFT as outlined in:
    //http://eelinux.ee.usm.maine.edu/courses/ele486/docs/makhoul.fastDCT.pdf   chapter: IV.B

    //1 - fill vreal array = resorted input
    resort2DArrayForward<<<FCT::gridSize, FCT::blockSize>>>(d_input, FCT::d_vreal, FCT::d_v_index, FCT::w, FCT::h); CUDA_CHECK;
    cudaDeviceSynchronize(); CUDA_CHECK;

    //2 - compute fft - real input vreal, complex output v
    cufftExecR2C(FCT::planf, FCT::d_vreal, FCT::d_v); CUDA_CHECK;
    cudaDeviceSynchronize(); CUDA_CHECK;

    ////3 - throw away unneeded FFT stuff, scale and fill output
    cuFCTCalcOutput<<<FCT::gridSize, FCT::blockSize>>>(FCT::d_v, d_output, FCT::d_roots, FCT::w, FCT::h); CUDA_CHECK;
    cudaDeviceSynchronize(); CUDA_CHECK;
  }

  ////
  //computes the Inverse Fast Discrete Cosine Transformation
  //
  //d_input, d_output = layered quadratic 2d w x h gpu array
  //requires gpu allocated memory + instance of FCT
  void FCT::ifct(float *d_input, float *d_output) {
    //http://eelinux.ee.usm.maine.edu/courses/ele486/docs/makhoul.fastDCT.pdf   chapter: IV.B

    //1 - undo scaling, reorder input into v
    cuIFCTPrepareInput<<<FCT::gridSize, FCT::blockSize>>>(d_input, FCT::d_v, FCT::d_roots, FCT::w, FCT::h); CUDA_CHECK;
    cudaDeviceSynchronize(); CUDA_CHECK;

    //2 - compute ifft - complex input v, real output vreal
    cufftExecC2R(FCT::plani, FCT::d_v, FCT::d_vreal); CUDA_CHECK;
    cudaDeviceSynchronize(); CUDA_CHECK;

    //3 - fill output array, throw away unneeded IFFT stuff
    resort2DArrayBackward<<<FCT::gridSize, FCT::blockSize>>>(FCT::d_vreal, d_output, FCT::d_v_index, FCT::w, FCT::h); CUDA_CHECK;
    cudaDeviceSynchronize(); CUDA_CHECK;
  }
}