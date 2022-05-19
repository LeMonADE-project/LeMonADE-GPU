[![DOI](https://zenodo.org/badge/136152127.svg)](https://zenodo.org/badge/latestdoi/136152127)
# LeMonADE-GPU
GPU extensions of the LeMonADE library
Repository with updaters, analyzers, and projects for sharing BFM stuff related to various topics.

## Installation

* Clone and Install `git clone https://github.com/LeMonADE-project/LeMonADE.git`
* Install cmake (minimum version 3.1)
* Install gcc   (minimum version 4.8)
* Install cuda  (minimum version 7.0)
* Just do for standard compilation:
 go to the main directory of the source code and do: 
````sh
    # generates the projects
    ./configure -DINSTALLDIR_LEMONADEGPU=/path/to/install/lemonadeGPU_install -DLEMONADE_DIR=/path/to/installation/lemonade_install  -DCUDA_ARCH=arch_of_graphics_card -DBUILDDIR=/path/to/build -DLEMONADEGPU_TESTS=ON/OFF -DCMAKE_BUILD_TYPE=Release/Debug
    make install
````

Compute capability can be found in : https://developer.nvidia.com/cuda-gpus


<!-- 
only important for local github server
## Note 

Unfortunately, the repo cannot be cloned from outside over ssh. Thus please use the https. For that one has to set 
> git config --global http.sslVerify true

A disadvantage is that for every pull/push, the username and password is requested.  -->

## License

See the LICENSE in the root directory.
