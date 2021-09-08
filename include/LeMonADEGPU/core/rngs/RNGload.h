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


#ifdef __CUDACC__
#   define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#   define CUDA_CALLABLE_MEMBER
#endif


namespace Rngs {


/* This is a class to load pregenerated RNGs */
class RNGload
{
public:
    using GlobalState = uint32_t;
    uint64_t miSubsequence;

    CUDA_CALLABLE_MEMBER inline  RNGload( void ){}
    CUDA_CALLABLE_MEMBER inline ~RNGload( void ){}

    CUDA_CALLABLE_MEMBER static constexpr bool needsSeed       ( void ){ return false; }
    CUDA_CALLABLE_MEMBER static constexpr bool needsSubsequence( void ){ return true ; }
    CUDA_CALLABLE_MEMBER static constexpr bool needsIteration  ( void ){ return false; }
    CUDA_CALLABLE_MEMBER static constexpr bool needsGlobalState( void ){ return true ; }

    CUDA_CALLABLE_MEMBER inline void setSeed       ( uint64_t const ){}
    CUDA_CALLABLE_MEMBER inline void setSubsequence( uint64_t const i ){ miSubsequence = i; }
    CUDA_CALLABLE_MEMBER inline void setIteration  ( uint64_t const ){}
    CUDA_CALLABLE_MEMBER inline void setGlobalState( GlobalState const * ptr ){ mPtr = ptr; }

    CUDA_CALLABLE_MEMBER inline uint32_t rng32( void ){ return mPtr[ miSubsequence ]; }

private:
    GlobalState const * mPtr;
};


}
