/*--------------------------------------------------------------------------------
    ooo      L   attice-based  |
  o\.|./o    e   xtensible     | LeMonADE: An Open Source Implementation of the
 o\.\|/./o   Mon te-Carlo      |           Bond-Fluctuation-Model for Polymers
oo--GPU--oo  A   lgorithm and  |
 o/./|\.\o   D   evelopment    | Copyright (C) 2013-2015 by
  o/.|.\o    E   nvironment    | LeMonADE Principal Developers (see AUTHORS)
    ooo                        |
----------------------------------------------------------------------------------

This file is part of LeMonADEGPU.

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
#pragma once

/**
 * These will be initialized to:
 *   DXTable_d = { -1,1,0,0,0,0 }
 *   DYTable_d = { 0,0,-1,1,0,0 }
 *   DZTable_d = { 0,0,0,0,-1,1 }
 * I.e. a table of three random directional 3D vectors \vec{dr} = (dx,dy,dz)
 */
__device__ __constant__ uint32_t DXTable_d[18]; //0:-x; 1:+x; 2:-y; 3:+y; 4:-z; 5+z
__device__ __constant__ uint32_t DYTable_d[18]; //0:-x; 1:+x; 2:-y; 3:+y; 4:-z; 5+z
__device__ __constant__ uint32_t DZTable_d[18]; //0:-x; 1:+x; 2:-y; 3:+y; 4:-z; 5+z
/* stores 2*dx in order to save the multiply with 2 operation ... */
__device__ __constant__ uint32_t DXTable2_d[18];
__device__ __constant__ uint32_t DYTable2_d[18];
__device__ __constant__ uint32_t DZTable2_d[18];
/**
 * If intCUDA is different from uint32_t, then this second table prevents
 * expensive type conversions, but both tables are still needed, the
 * uint32_t version, because the calculation of the linear index will result
 * in uint32_t anyway and the intCUDA version for solely updating the
 * position information
 * problematic to templatize ...
 * @see https://stackoverflow.com/questions/25008402/cuda-declaring-a-device-constant-as-a-template
 *   => seems like he does / I have to manage the constant memory myself
 *      problem would appear if multiple kernels with differing types are
 *      running... or rather different typed ::initialize members are called,
 *      as then we the constant memroy data would hold the wrong value!
 *      Working with offsets depending on the type might work, i.e. modify
 *      the *( (int*) data ) to something like return *( (T*)( (T_LargestType*) data + sizeof(T) / sizeof( T_SmallestType ) - 1 )
 *      this should work assuming
 * @todo try again with Pascal
 */
/*
__device__ __constant__ UpdaterGPUScBFM_AB_Type::T_UCoordinateCuda DXTableUintCuda_d[6];
__device__ __constant__ UpdaterGPUScBFM_AB_Type::T_UCoordinateCuda DYTableUintCuda_d[6];
__device__ __constant__ UpdaterGPUScBFM_AB_Type::T_UCoordinateCuda DZTableUintCuda_d[6];
*/

/* will this really bring performance improvement? At least constant cache
 * might be as fast as register access when all threads in a warp access the
 * the same constant */
__device__ __constant__ uint32_t dcBoxX     ;  // lattice size in X
__device__ __constant__ uint32_t dcBoxY     ;  // lattice size in Y
__device__ __constant__ uint32_t dcBoxZ     ;  // lattice size in Z
__device__ __constant__ uint32_t dcBoxXM1   ;  // lattice size in X-1
__device__ __constant__ uint32_t dcBoxYM1   ;  // lattice size in Y-1
__device__ __constant__ uint32_t dcBoxZM1   ;  // lattice size in Z-1
__device__ __constant__ uint32_t dcBoxXLog2 ;  // lattice shift in X
__device__ __constant__ uint32_t dcBoxXYLog2;  // lattice shift in X*Y

