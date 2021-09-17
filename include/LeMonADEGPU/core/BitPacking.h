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
#ifndef LEMONADEGPU_CORE_BITPACKING_H
#define  LEMONADEGPU_CORE_BITPACKING_H
#include <stdio.h>
class BitPacking {

public:
  BitPacking():bitPackingOn(false),nBbufferedTmpLatticeOn(false){};
  BitPacking(bool bitPackingOn_):bitPackingOn(bitPackingOn_),nBbufferedTmpLatticeOn(false){};
  BitPacking(bool bitPackingOn_, bool nBbufferedTmpLatticeOn_):bitPackingOn(bitPackingOn_),nBbufferedTmpLatticeOn(nBbufferedTmpLatticeOn_){};
//   ~BitPacking(){};
  
  __host__ inline void setBitPackingOn(bool bitPackingOn_){bitPackingOn=bitPackingOn_;}
   inline bool getBitPackingOn()const {return bitPackingOn;}
  
  __host__ inline void setNBufferedTmpLatticeOn(bool nBbufferedTmpLatticeOn_){nBbufferedTmpLatticeOn=nBbufferedTmpLatticeOn_;}
   inline bool getNBufferedTmpLatticeOn()const {return nBbufferedTmpLatticeOn;}
  
  template< typename T, typename T_Id > __device__ __host__ inline
  T bitPackedGet( T const * const & p, T_Id const & i );
  
  template< typename T > __device__  inline 
  T bitPackedTextureGet( cudaTextureObject_t p, int i ) const ;
  
  template< typename T > __device__  inline 
  T bitPackedTextureGetStandard( cudaTextureObject_t p, int i ) const  
  {
    //I just dont know why I have to use this macro in the following, 
    //but without a get a declaration error which says that the compiler find no for tex1Dfetch
  #ifdef __CUDA_ARCH__ 
    return tex1Dfetch<T>(p,i); 
  #endif
  }

  /**
    * Because the smalles atomic is for int (4x uint8_t) we need to
    * cast the array to that and then do a bitpacking for the whole 32 bits
    * instead of 8 bits
    * I.e. we need to address 32 subbits, i.e. >>3 becomes >>5
    * and &7 becomes &31 = 0b11111 = 0x1F
    * __host__ __device__ function with differing code
    * @see https://codeyarns.com/2011/03/14/cuda-common-function-for-both-host-and-device-code/
    */
  template< typename T, typename T_Id > __device__ __host__ inline
  void bitPackedSet( T * const __restrict__ p, T_Id const & i ) const ;
  template< typename T, typename T_Id > __device__ __host__ inline
  void bitPackedUnset( T * const __restrict__ p, T_Id const & i );
  
  
private:
  
  bool bitPackingOn;
  bool nBbufferedTmpLatticeOn;
};

template< typename T, typename T_Id > __device__ __host__ inline
T BitPacking::bitPackedGet( T const * const & p, T_Id const & i )
{
    /**
      * >> 3, because 3 bits = 2^3=8 numbers are used for sub-byte indexing,
      * i.e. we divide the index i by 8 which is equal to the space we save
      * by bitpacking.
      * & 7, because 7 = 0b111, i.e. we are only interested in the last 3
      * bits specifying which subbyte element we want
      */
    switch (bitPackingOn) {
      case 0:  return p[i];
      case 1:  return ( p[ i >> 3 ] >> ( i & T_Id(7) ) ) & T(1);
    };
}

template< typename T > __device__  inline
T BitPacking::bitPackedTextureGet( cudaTextureObject_t p, int i )  const 
{
  //I just dont know why I have to use this macro in the following, 
  //but without a get a declaration error which says that the compiler find no for tex1Dfetch
#ifdef __CUDA_ARCH__ 
  switch (bitPackingOn) {
    case 0 : return bitPackedTextureGetStandard<T>(p,i); 
    case 1 : return ( tex1Dfetch<T>( p, i >> 3 ) >> ( i & 7 ) ) & T(1);
  };
#endif
  return T();
}

// template  __device__   uint8_t BitPacking::bitPackedTextureGet ( cudaTextureObject_t, int ) const  ;


/**
  * Because the smalles atomic is for int (4x uint8_t) we need to
  * cast the array to that and then do a bitpacking for the whole 32 bits
  * instead of 8 bits
  * I.e. we need to address 32 subbits, i.e. >>3 becomes >>5
  * and &7 becomes &31 = 0b11111 = 0x1F
  * __host__ __device__ function with differing code
  * @see https://codeyarns.com/2011/03/14/cuda-common-function-for-both-host-and-device-code/
  */
template< typename T, typename T_Id > __device__ __host__ inline 
void BitPacking::bitPackedSet( T * const __restrict__ p, T_Id const & i ) const 
{
  switch (bitPackingOn) {
    case 0 :  p[i] = 1;
    case 1 :  static_assert( sizeof(int) == 4, "" );
	      #ifdef __CUDA_ARCH__
		  atomicOr ( (int*) p + ( i >> 5 ),    T(1) << ( i & T_Id( 0x1F ) )   );
	      #else
		  p[ i >> 3 ] |= T(1) << ( i & T_Id(7) );
	      #endif
  };
}

template< typename T, typename T_Id > __device__ __host__ inline
void BitPacking::bitPackedUnset( T * const __restrict__ p, T_Id const & i )
{
  switch (bitPackingOn) {
    case 0 :  p[i] = 0;
    case 1 :
	      #ifdef __CUDA_ARCH__
		  atomicAnd( (uint32_t*) p + ( i >> 5 ), ~( uint32_t(1) << ( i & T_Id( 0x1F ) ) ) );
	      #else
		  p[ i >> 3 ] &= ~( T(1) << ( i & T_Id(7) ) );
	      #endif
  };
} 
  

#endif