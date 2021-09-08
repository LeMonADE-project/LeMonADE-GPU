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

#include <stdint.h>
//#include <curand.h>
#include <curand_kernel.h>


#ifdef __CUDACC__
#   define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#   define CUDA_CALLABLE_MEMBER
#endif


namespace Rngs {


class lemonade_philox
{
private:
    uint64_t mSeed;
    uint64_t mIteration;
    uint64_t mSubseq;

    curandStatePhilox4_32_10_t c_state;
    //curandState c_state;
    bool initialized;

public:
    //Int instead of void to suppress compiler warnings
    using GlobalState = int;


    CUDA_CALLABLE_MEMBER inline  lemonade_philox( void ) : initialized( false ){}
    CUDA_CALLABLE_MEMBER inline ~lemonade_philox( void ){};

    CUDA_CALLABLE_MEMBER static constexpr bool needsSeed       ( void ){ return true ; }
    CUDA_CALLABLE_MEMBER static constexpr bool needsSubsequence( void ){ return true ; }
    CUDA_CALLABLE_MEMBER static constexpr bool needsIteration  ( void ){ return true ; }
    CUDA_CALLABLE_MEMBER static constexpr bool needsGlobalState( void ){ return false; }

    CUDA_CALLABLE_MEMBER inline void setSeed       ( uint64_t const rSeed      ){ mSeed      = rSeed     ; }
    CUDA_CALLABLE_MEMBER inline void setSubsequence( uint64_t const rSubseq    ){ mSubseq    = rSubseq   ; }
    CUDA_CALLABLE_MEMBER inline void setIteration  ( uint64_t const rIteration ){ mIteration = rIteration; }
    CUDA_CALLABLE_MEMBER inline void setGlobalState( void const * ){}

    //CUDA_CALLABLE_MEMBER uint32_t rng32(void)
    __device__ inline uint32_t rng32( void )
    {
        if ( ! initialized  )
        {
            curand_init( mSeed, mSubseq, mIteration, &c_state );
            initialized = true;
        }
        return curand( &c_state );
    }
};


}
