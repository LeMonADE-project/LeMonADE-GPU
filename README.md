# LeMonADE-GPU
GPU extensions of the LeMonADE library
Repository with updaters, analyzers, and projects for sharing BFM stuff related to various topics.

## Installation

* Clone and Install `git clone https://github.com/LeMonADE-project/LeMonADE.git`
* Install cmake (minimum version 3.1)
* Install gcc   (minimum version 4.8)
* Install cuda  (minimum version 7.0)
* Just do for standard compilation:
 
````sh
    # generates the projects
    mkdir build
    cd build
    ./configure -DINSTALLDIR_LEMONADEGPU=/path/to/install/lemonadeGPU_install -DLEMONADE_DIR=/path/to/installation/lemonade_install  -DCUDA_ARCH=60 -DBUILDDIR=/path/to/build -DLEMONADEGPU_TESTS=ON/OFF -DCMAKE_BUILD_TYPE=Release/Debug
    make -j 2 
````

## License

See the LICENSE in the root directory.
