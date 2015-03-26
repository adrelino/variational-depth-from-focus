variational-depth-from-focus
============================

CUDA implementation of: 
Moeller, Michael, et al. [Variational Depth from Focus Reconstruction](http://arxiv.org/pdf/1408.0173v2.pdf). arXiv preprint arXiv:1408.0173v2 (2014).

>This paper deals with the problem of reconstructing a depth map from a sequence of differently focused images, also known as depth from focus or shape from focus. We propose to state the depth from focus problem as a variational problem including a smooth but nonconvex data fidelity term, and a convex nonsmooth regularization, which makes the method robust to noise and leads to more realistic depth maps. Additionally, we propose to solve the nonconvex minimization problem with a linearized alternating directions method of multipliers (ADMM), allowing to minimize the energy very efficiently. A numerical comparison to classical methods on simulated as well as on real data is presented.

### Dependencies 
CMake, CUDA, OpenCV

#### MacOSX (10.9, 10.10)
We installed OpenCV 2.4.10 and CMake 3 using brew

make sure brew is installed correctly
```sh
ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
brew doctor
```

install cmake and opencv
```sh
brew install cmake
brew tap homebrew/science
brew install opencv
```

Install CUDA 7.0, which comes with C++11 support and is compiled with libc++ which is the default on OSX since 10.9. This really simplifies the installation, because prior to 7.0, CUDA was built with libstdc++, so the OpenCV dependency would have to be built with libstdc++ as well (e.g. by passing the --with-cuda flag).

http://developer.download.nvidia.com/compute/cuda/7_0/Prod/local_installers/cuda_7.0.29_mac.pkg


#### Linux
Our code was tested under Arch Linux with OpenCV 2.4.10, CMake 3.2.1 and CUDA 7.0.
It should compile and run successfully under your favorite linux distribution if it provides the above mentioned dependencies.

Please use the appropriate package manager of your distribution to install these packages; below is an example
for Arch Linux

```sh
pacman -S opencv
pacman -S cmake
pacman -S cuda
```

#### Windows
The code was tested under Windows 8.1, OpenCV 2.4.10, CMake 3.2.1, CUDA 7.0 and assumes you have a working version of Visual Studio installed.In our case, we worked with Visual Studio Ultimate 2013. 

If you do not have a version of Visual Studio, it is recommended that you install it first.
One can obtain a free version under the following link:
https://www.visualstudio.com/en-us/products/visual-studio-express-vs.aspx

Download and install CUDA 7.0, which can be found under:  
https://developer.nvidia.com/cuda-downloads  
  
Next, install OpenCV 2.4.10 from  
https://sourceforge.net/projects/opencvlibrary/files/opencv-win/2.4.10/opencv-2.4.10.exe/download  
Please remember in which path you installed OpenCV, since we have to add to the Windows PATH environment variable
```
\path\to\opencv\build\x64\vc12\bin
```
If necessary, replace x64 with x86 for a 32-bit system and vc12 with vc11 for Visual Studio 12 or vc10 for Visual Studio 10.  
  
Lastly, we need to install CMake, which can be obtained under:
http://www.cmake.org/download/

### Installation
Now we are ready to build our code

#### Mac OSX and Linux
Check out our repository, create a separate build-folder and then build the files using CMake.

```sh
cd ~/projects
git clone https://github.com/adrelino/variational-depth-from-focus.git
cd variational-depth-from-focus
mkdir build && cd build
cmake ..
make
```
#### Windows 
Check out the repository under some directory, e.g. 
```
C:\variational-depth-from-focus
```
Start up CMake and
1. specify where the source is. In our case it would be under C:\variational-depth-from-focus
### Execution

#### Run with included synthetic sample sequence
```sh
./main
```

#### More sequences on real data
* http://www.sayonics.com/sources/books_00.zip
* http://www.sayonics.com/sources/books_02.zip
* http://www.sayonics.com/sources/books_05.zip

unzip them and then run with the -dir option:
```sh
./main -dir <DIR>
```
