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
#ifndef LEMONADEGPU_FEATURE_CHECK_DENSITY_H
#define LEMONADEGPU_FEATURE_CHECK_DENSITY_H
#include <LeMonADEGPU/utility/cudacommon.hpp>
#include <LeMonADEGPU/utility/MirroredVector.h>
#include <cuda_profiler_api.h>              // cudaProfilerStop
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
typedef uint8_t intArray;
typedef uint32_t boxType;
////////////////////////////////////////////////////////////////////////////////
//device constants 
//! average number of monomers in 4 slices in the middle and in the boundaries
__device__ __constant__ uint32_t dAvMonomerNumberInShearVolume; 
__device__ __constant__ uint32_t dMonomerNumber_in_ShearVolumeMiddle;
__device__ __constant__ uint32_t dMonomerNumber_in_ShearVolumeBoundary;
__device__ __constant__ bool     dIsOn;
__device__ __constant__ boxType  dBoxX;
__device__ __constant__ boxType  dBoxY;
__device__ __constant__ boxType  dBoxZ;
//d-device 
//M-minus
//P-plu
//h-half
__device__ __constant__ boxType dBoxZM1;
__device__ __constant__ boxType dBoxZM2;
__device__ __constant__ boxType dBoxZM3;
__device__ __constant__ boxType dBoxZhP1;
__device__ __constant__ boxType dBoxZhP2;
__device__ __constant__ boxType dBoxZhM3;
__device__ __constant__ boxType dBoxZhM2;
// __device__ __constant__ boxType dBoxZhP3;
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
/**
 * @file checkDensity.h 
 * @class checkDensity
 * @brief used to get a constant density during shear
 * @details In the shear simulations a force in x-direction is applied in two 
 * layers at the top and the bottom and in opposite direction in four layers in 
 * the middle of the layer. For same cases therer is a reduced or enhanced 
 * density inside these layers, which we would identify as an artefact. Thus, there
 * is a need to held the number of monomers close to the average number to ensure 
 * a useful simulation.
 * This class uses a reduction algorithm which ist explained in 
 * https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
 */
template <typename T_UCoordinateCuda > class checkDensity
{
public:
    //type for unsigned coordinates on the device 
    using T_UCoordinatesCuda = typename CudaVec4< T_UCoordinateCuda >::value_type;
	/** constructor 
	 */
    checkDensity();
	checkDensity(uint32_t BoxX_, uint32_t BoxY_, uint32_t BoxZ_);
    ~checkDensity();
	/**
	 * 
	 */
    
    __device__ inline bool operator()( uint8_t const  z,  int32_t const dz) const 
    {
        if ( !dIsOn) return true; 
        if (dz == 0) return true;  
        // auto z=z0%dBoxZ;
        // auto z=z0%BoxZ;
        //monomers moves up 
        // printf("z=%d z0=%d dz=%d\n",z,z0,dz );
        // printf("box=(%d,%d,%d)\n",dBoxX, dBoxY, dBoxZ);
        // printf("%d %d %d %d %d %d %d %d \n",dBoxZM1, dBoxZM2, dBoxZM3, dBoxZhM2, dBoxZhM3, dBoxZhP1, dBoxZhP2, dBoxZhP3);
        // printf("op(): %d %d %d %d \n", dAvMonomerNumberInShearVolume,dMonomerNumber_in_ShearVolumeBoundary,dMonomerNumber_in_ShearVolumeMiddle, dBoxZ);
        /**
         * Make an example, to choose the correct boundaries!
         * BoxSize=16
         * 00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 (0)
         *  x  x  0  0  0  0  x  x  x  x  0  0  0  0  x  x 
         * x refers to where the force is applied and where the 
         * density needs to keept constant.
         * dBoxZM3 =13 
         * dBoxZM2 =14
         * dBoxZhM3=5
         * dBoxZhM2=6
         * dBoxZhP1=9
         * dBoxZhP2=10
         */
        if(dz == 1){
            //density in the lowest layer is smaller than the average 
            if(dMonomerNumber_in_ShearVolumeBoundary < dAvMonomerNumberInShearVolume ){ 
                if(z == 1) return false ;
            }else{ //density in toplayer is greater than the average 
                if(z == dBoxZM3) return false ;
            }
            //density in the middle layer is smaller than the average 
            if(dMonomerNumber_in_ShearVolumeMiddle < dAvMonomerNumberInShearVolume){
                if(z == dBoxZhP1) return false ;
            }else { //density in middle is greater than the average 
                if (z == dBoxZhM3) return false ;
            }
        }else if(dz == -1){
            if(dMonomerNumber_in_ShearVolumeBoundary < dAvMonomerNumberInShearVolume ){ 
                if(z == dBoxZM2) return false ;
            }else{ 
                if(z == 2) return false ;
            }
            if(dMonomerNumber_in_ShearVolumeMiddle < dAvMonomerNumberInShearVolume){
                if(z == dBoxZhM2 )return false ;
            }else{
                if (z == dBoxZhP2) return false ;
            }
        }
        return true;
    };
    //!check the correct settin of the constants
    void launch_checkConstantSetting();
    void printResults();
    //!
    void launch_countMonomers( 
        MirroredVector<T_UCoordinatesCuda> * mpPolymerSystem,
        uint32_t offset,
        size_t nMons,
        size_t const nBlocksSpecies,
        uint32_t const nThreadsSprecies );
	//! 
    void init(uint32_t NMonomers_, uint32_t nSortedMonomers, cudaDeviceProp&  mCudaProps);
	//! 
    void calcDensity(void); 
	//! 
	void setBoxSizes(uint32_t BoxX_, uint32_t BoxY_, uint32_t BoxZ_);
    //!
    void setDensityCheckON(bool checkOn_);
private:
	//! count the number of monomers in the middle
	MirroredVector<intArray>*  mCountMiddleMonos;
	//! count the number of monomers in the boundary
	MirroredVector<intArray> * mCountBoundaryMonos;
	//!array after the reduction used primarily for the result: middle monomers 
	MirroredVector<intArray> * mReducedCountMiddleMonos;
	//!array after the reduction used primarily for the result: boundary monomers
	MirroredVector<intArray> * mReducedCountBoundaryMonos;
	//! number of threads 
	uint32_t threads;
	//! size of the mirrored vectors 
	uint32_t arraySize;
    //!
    uint32_t hAvMonomerNumberInShearVolume;
	//! number of 
	uint32_t hMonomerNumber_in_ShearVolumeMiddle;
	//!
	uint32_t hMonomerNumber_in_ShearVolumeBoundary;
    // //!
    // uint32_t * dAvMonomerNumberInShearVolume;
	// //!
	// uint32_t * dMonomerNumber_in_ShearVolumeMiddle;
	// //!
	// uint32_t * dMonomerNumber_in_ShearVolumeBoundary;
	//! box sizes in the three directions
	boxType BoxX,BoxY,BoxZ; 
    //! number of monomers in the simulation box 
    uint32_t NMonomers;
    //! set the check on or off
    bool checkOn;
	//! number of blocks for the kernel which performs the reduction
	uint32_t nBlocks;
	//! number of threads for the kernel which performs the reduction
	uint32_t nThreads;
};
#endif /*LEMONADEGPU_FEATURE_CHECK_DENSITY_H*/