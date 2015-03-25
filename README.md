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
TODO


#### Windows
TODO


### Installation
Now we are ready to build our code
```sh
cd ~/projects
git clone https://github.com/adrelino/variational-depth-from-focus.git
cd variational-depth-from-focus
mkdir build && cd build
cmake ..
make
```


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