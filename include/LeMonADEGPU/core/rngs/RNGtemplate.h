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
#ifndef RNG_TEMPLATE_H
#define RNG_TEMPLATE_H

#include <cstdint>

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

//This is just a template class, never use it
class templateRNG{
public:
    //Int instead of void to suppress compiler warnings
    typedef int global_state_type;

    CUDA_CALLABLE_MEMBER templateRNG(void)
        {}

    CUDA_CALLABLE_MEMBER ~templateRNG(void)
        {}

    CUDA_CALLABLE_MEMBER static constexpr bool needs_global_state(void)
        {return false;}

    CUDA_CALLABLE_MEMBER void set_global_state(const global_state_type*ptr)
        {}

    CUDA_CALLABLE_MEMBER static constexpr bool needs_iteration(void)
        {return false;}

    CUDA_CALLABLE_MEMBER void set_iteration(const uint64_t iteration)
        {}

    CUDA_CALLABLE_MEMBER static constexpr bool needs_seed(void)
        {return false;}

    CUDA_CALLABLE_MEMBER void set_seed(const uint64_t seed)
        {}

    CUDA_CALLABLE_MEMBER static constexpr bool needs_subsequence(void)
        {return false;}

    CUDA_CALLABLE_MEMBER void set_subsequence(const uint64_t subseq)
        {}

    CUDA_CALLABLE_MEMBER uint32_t rng32(void)
        {return 0;}

    };

#endif//RNG_TEMPLATE_H
