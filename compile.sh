#! /bin/bash 
rm -r build/
./configure -DINSTALLDIR_LEMONADEGPU=/home/Toe-Knee/software/lemonadeGPU_install -DLEMONADE_DIR=/home/Toe-Knee/software/lemonade_install  -DCUDA_ARCH=60
make -j 2
