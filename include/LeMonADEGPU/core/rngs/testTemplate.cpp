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
#include "Hash.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

#include "PCG.h"


uint32_t hashLEMONADE( uint32_t a )
{
    a = ( a + 0x7ed55d16 ) + ( a << 12 );
    a = ( a ^ 0xc761c23c ) ^ ( a >> 19 );
    a = ( a + 0x165667b1 ) + ( a << 5  );
    a = ( a + 0xd3a2646c ) ^ ( a << 9  );
    a = ( a + 0xfd7046c5 ) + ( a << 3  );
    a = ( a ^ 0xb55a4f09 ) ^ ( a >> 16 );
    return a;
}


template< typename RNG >
void kernel
(
    unsigned int const seed,
    unsigned int const iteration,
    typename RNG::global_state_type * global_state_array,
    unsigned int const Nmonomers)
{
    for( auto mono = 0u; mono < Nmonomers; ++mono )
    {
        RNG rng;

        if( RNG::needsGlobalState() ) rng.setGlobalState( global_state_array+mono );
        if( RNG::needsIteration  () ) rng.setIteration  ( iteration );
        if( RNG::needsSubsequence() ) rng.setSubsequence( mono      );
        if( RNG::needsSeed       () ) rng.setSeed       ( seed      );

        std::cout << rng.rng32() << " " << rng.rng32() << " " << rng.rng32() << std::endl;
    }
}


int main(int argc,char*argv[])
{
    auto const Nsteps    = 12000;
    auto const seed      = 42   ;
    auto const Nmonomers = 123  ;

    std::vector< PCG::State > state_vector( Nmonomers );
    for ( auto i = 0u; i < Nmonomers; ++i )
        state_vector[i] = PCG::State( seed, i );
    PCG::global_state_type* global_state_array = state_vector.data();

    for( auto global_step = 0; global_step < Nsteps; ++global_step )
        kernel<PCG>(seed,global_step,global_state_array,Nmonomers);

    return 0;
}
