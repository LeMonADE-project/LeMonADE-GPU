/*--------------------------------------------------------------------------------
    ooo      L   attice-based  |
  o\.|./o    e   xtensible     | LeMonADE: An Open Source Implementation of the
 o\.\|/./o   Mon te-Carlo      |           Bond-Fluctuation-Model for Polymers
oo---0---oo  A   lgorithm and  |
 o/./|\.\o   D   evelopment    | Copyright (C) 2013-2015 by
  o/.|.\o    E   nvironment    | LeMonADE Principal Developers (see AUTHORS)
    ooo                        |
----------------------------------------------------------------------------------

This file is part of LeMonADE.

LeMonADE is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

LeMonADE is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with LeMonADE.  If not, see <http://www.gnu.org/licenses/>.

--------------------------------------------------------------------------------*/
#ifndef LEMONADE_CORE_SPACE_FILLING_CURVE_CU
#define LEMONADE_CORE_SPACE_FILLING_CURVE_CU

#include <extern/Fundamental/BitsCompileTime.hpp>
#include <LeMonADEGPU/core/SpaceFillingCurve.h>
#include <LeMonADEGPU/utility/cudacommon.hpp>
#include <stdio.h>

/*****************************************************************************/
/**
 *@brief abstract template function for the space filling curve 
 */
/*****************************************************************************/
 __constant__ size_t dcBoxX     ;  // lattice size in X
 __constant__ size_t dcBoxY     ;  // lattice size in Y
 __constant__ size_t dcBoxZ     ;  // lattice size in Z
 __constant__ size_t dcBoxXM1   ;  // lattice size in X-1
 __constant__ size_t dcBoxYM1   ;  // lattice size in Y-1
 __constant__ size_t dcBoxZM1   ;  // lattice size in Z-1
 __constant__ size_t dcBoxXLog2 ;  // lattice shift in X
 __constant__ size_t dcBoxXYLog2;  // lattice shift in X*Y
__global__ void CheckBoxDimensionsSpaceFilling()
{
printf("CheckBoxDimensionsSpaceFilling: %lu %lu %lu %lu %lu %lu %lu %lu\n",dcBoxX,dcBoxY, dcBoxZ,dcBoxXM1, dcBoxYM1,dcBoxZM1, dcBoxXLog2, dcBoxXYLog2 );
}
SpaceFillingCurve::SpaceFillingCurve(){};

SpaceFillingCurve::SpaceFillingCurve(uint64_t mBoxX_, uint64_t mBoxY_, uint64_t mBoxZ_, int mode )
{
    setBox(mBoxX_,mBoxY_,mBoxZ_);
}

void SpaceFillingCurve::setBox(uint64_t mBoxX_, uint64_t mBoxY_, uint64_t mBoxZ_)
{
    { decltype( dcBoxX      ) x = mBoxX_     ; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxX     , &x, sizeof(x) ) ); }
    { decltype( dcBoxY      ) x = mBoxY_     ; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxY     , &x, sizeof(x) ) ); }
    { decltype( dcBoxZ      ) x = mBoxZ_     ; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxZ     , &x, sizeof(x) ) ); }
    { decltype( dcBoxXM1    ) x = mBoxX_-1   ; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxXM1   , &x, sizeof(x) ) ); }
    { decltype( dcBoxYM1    ) x = mBoxY_-1   ; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxYM1   , &x, sizeof(x) ) ); }
    { decltype( dcBoxZM1    ) x = mBoxZ_-1   ; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxZM1   , &x, sizeof(x) ) ); }

    //differentiate between the different curves by the mode type: 
    uint64_t mBoxXLog2(0), mBoxXYLog2(0);
    switch( mode )
    {
        case 2: lP2Curve.initialize(mBoxX_,mBoxY_,mBoxZ_);
                {auto dummy = mBoxX_ ; while ( dummy >>= 1 ) ++mBoxXLog2;
                dummy = mBoxX_*mBoxY_; while ( dummy >>= 1 ) ++mBoxXYLog2;}
                { decltype( dcBoxXLog2  ) x = mBoxXLog2  ; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxXLog2 , &x, sizeof(x) ) ); }
                { decltype( dcBoxXYLog2 ) x = mBoxXYLog2 ; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxXYLog2, &x, sizeof(x) ) ); } 
                break;
        case 1: lCurve.initialize(mBoxX_,mBoxY_,mBoxZ_);
                break;
        case 0: zCurve.initialize(mBoxX_,mBoxY_,mBoxZ_);
                {auto dummy = mBoxX_ ; while ( dummy >>= 1 ) ++mBoxXLog2;
                dummy = mBoxX_*mBoxY_; while ( dummy >>= 1 ) ++mBoxXYLog2;}
                { decltype( dcBoxXLog2  ) x = mBoxXLog2  ; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxXLog2 , &x, sizeof(x) ) ); }
                { decltype( dcBoxXYLog2 ) x = mBoxXYLog2 ; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxXYLog2, &x, sizeof(x) ) ); } 
                break;
    }
    CheckBoxDimensionsSpaceFilling<<<1,1>>>();
}

int SpaceFillingCurve::getMode() const {return mode;}
void SpaceFillingCurve::setMode( int mode_) { mode=mode_; }
 
#endif 
