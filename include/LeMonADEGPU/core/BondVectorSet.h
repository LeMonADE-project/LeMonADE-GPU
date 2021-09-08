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
#ifndef LEMONADEGPU_CORE_BONDVECTORSET_H_
#define LEMONADEGPU_CORE_BONDVECTORSET_H_
#include <LeMonADEGPU/utility/cudacommon.hpp> //for  CUDA_ERROR
/* 512=8^3 for a range of bonds per direction of [-4,3] */
__device__ __constant__ bool dpForbiddenBonds[512]; //false-allowed; true-forbidden
/**
 * @brief I declared this function globally otherwise i decrease the performance by around 200 % 
 * 	  unfortunatly I have no idea why?!
 */
__device__ __host__ int16_t inline  linearizeBondVectorIndex
(
    int16_t const x,
    int16_t const y,
    int16_t const z
) {
      /* Just like for normal integers we clip the range to go more down than up
  * i.e. [-127 ,128] or in this case [-4,3]
  * +4 maps to the same location as -4 but is needed or else forbidden
  * bonds couldn't be detected. Larger bonds are not possible, because
  * monomers only move by 1 per step */
  //assert( -4 <= x && x <= 4 );
  //assert( -4 <= y && y <= 4 );
  //assert( -4 <= z && z <= 4 );
  return   ( x & int16_t(7) /* 0b111 */ ) +
	( ( y & int16_t(7) /* 0b111 */ ) << 3 ) +
	( ( z & int16_t(7) /* 0b111 */ ) << 6 );
}

/**
 * @brief simple class to decide about acceptable bonds
 * @details in the standard construction of the program this operator returns true for a bond which is NOT valid 
 * and the values for dx,dy,dz range from [-4:3]. Other values cannot be handled and thus cann cause serious errors!
 * @todo add a selectiveLogger to the class which performs a check for the range if the "check" case is activated 
 */

class BondVectorSet
{
public:


  inline void addBondVector( int dx, int dy, int dz, bool bondForbidden ){mForbiddenBonds[ linearizeBondVectorIndex(dx,dy,dz) ] = bondForbidden;};

  /**  
   * @brief checks if the bond is forbidden and returns true if so 
   * @details If differenceGreaterFour is turned true, the difference in each 
   * direction is checked if it exceeds the absolute of 4 which is not allowed.
   * @param dx difference in x direction
   * @param dy difference in y direction
   * @param dz difference in z direction
   * @param differenceGreaterFour is a template parameter with default false
   * 
   */
//   template <bool differenceGreaterFour=false >
  __device__ __host__ inline   bool operator()(int dx, int dy, int dz) const 
  {

//     if ( differenceGreaterFour == true ) 
//     {
//       if(dx*dx > 9) return true; 
//       if(dy*dy > 9) return true; 
//       if(dz*dz > 9) return true; 
//     }
    //if check activated-> check if dx,dy,dz are in the correct range [-4:3]
    #ifdef __CUDA_ACC__
      return dpForbiddenBonds[linearizeBondVectorIndex(dx,dy,dz)];
    #else
      return mForbiddenBonds[linearizeBondVectorIndex(dx,dy,dz)];
    #endif
  }

  void initBondTable();
private:   
  bool mForbiddenBonds[512];  
  
};



#endif /* LEMONADEGPU_CORE_BONDVECTORSET_H_ */
