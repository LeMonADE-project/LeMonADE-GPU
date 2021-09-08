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


/* This is the original hash based class, the quality is in doubt. */
class Hash
{
private:
    uint32_t mState;
    uint32_t mSeed;

    //change this function if neccessary
    CUDA_CALLABLE_MEMBER inline static uint32_t hash( uint32_t a )
    {
        /* https://web.archive.org/web/20120626084524/http://www.concentric.net:80/~ttwang/tech/inthash.htm
         * Note that before this 2007-03 version there were no magic numbers.
         * This hash function doesn't seem to be published.
         * He writes himself that this shouldn't really be used for PRNGs ???
         * @todo E.g. check random distribution of randomly drawn directions are
         *       they rouhgly even?
         * The 'hash' or at least an older version of it can even be inverted !!!
         * http://c42f.github.io/2015/09/21/inverting-32-bit-wang-hash.html
         * Somehow this also gets attibuted to Robert Jenkins?
         * https://gist.github.com/badboy/6267743
         * -> http://www.burtleburtle.net/bob/hash/doobs.html
         *    http://burtleburtle.net/bob/hash/integer.html
         */
        a = ( a + 0x7ed55d16 ) + ( a << 12 );
        a = ( a ^ 0xc761c23c ) ^ ( a >> 19 );
        a = ( a + 0x165667b1 ) + ( a << 5  );
        a = ( a + 0xd3a2646c ) ^ ( a << 9  );
        a = ( a + 0xfd7046c5 ) + ( a << 3  );
        a = ( a ^ 0xb55a4f09 ) ^ ( a >> 16 );
        return a;
    }

public:
    //Int instead of void to suppress compiler warnings
    using GlobalState = int;

    CUDA_CALLABLE_MEMBER inline  Hash( void ){}
    CUDA_CALLABLE_MEMBER inline ~Hash( void ){}

    CUDA_CALLABLE_MEMBER static constexpr bool needsSeed       ( void ){ return true ; }
    CUDA_CALLABLE_MEMBER static constexpr bool needsSubsequence( void ){ return true ; }
    CUDA_CALLABLE_MEMBER static constexpr bool needsIteration  ( void ){ return false; }
    CUDA_CALLABLE_MEMBER static constexpr bool needsGlobalState( void ){ return false; }

    CUDA_CALLABLE_MEMBER inline void setSeed       ( uint64_t const rSeed  ){ mSeed  = rSeed ; }
    CUDA_CALLABLE_MEMBER inline void setSubsequence( uint64_t const subseq ){ mState = subseq; }
    CUDA_CALLABLE_MEMBER inline void setIteration  ( uint64_t const ){}
    CUDA_CALLABLE_MEMBER inline void setGlobalState( void const * ){}

    CUDA_CALLABLE_MEMBER inline uint32_t rng32( void )
    {
        mState = hash( hash( mState ) ^ mSeed );
        return mState;
    }
};


}
