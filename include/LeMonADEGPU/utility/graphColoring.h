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

#ifndef INLINE_GRAPHCOLORING
#   define INLINE_GRAPHCOLORING inline
#endif

#include <cstdint>                      // uint8_t
#include <functional>
#include <vector>


template< class T_Neighbors, typename T_Id = size_t, typename T_Color = uint8_t >
INLINE_GRAPHCOLORING
std::vector< T_Color > graphColoring
(
    T_Neighbors const & rvNeighbors    ,
    size_t      const & rnElements     ,
    bool        const   rbUniformColors,
    std::function< size_t( T_Neighbors const &, T_Id const & ) > const & rfGetNeighborsSize,
    /* lambdas as default arguments gives errors for gcc 4.9.1 and below and works for 4.9.3 or greater */
   /* 
    #if defined ( __GNUC__ ) && ( __GNUC__ >= 5 || ( __GNUC__ == 4 && ( __GNUC_MINOR__ >= 10 || ( __GNUC_MINOR__ == 9 && __GNUC_PATCHLEVEL__ >= 3 ) ) ) )
        = []( T_Neighbors const & x, T_Id const & i ){ return x[i].size(); },
    #endif
    */
    std::function< T_Id( T_Neighbors const &, T_Id const &, size_t const & ) > const & rfGetNeighbor
    /*
    #if defined ( __GNUC__ ) && ( __GNUC__ >= 5 || ( __GNUC__ == 4 && ( __GNUC_MINOR__ >= 10 || ( __GNUC_MINOR__ == 9 && __GNUC_PATCHLEVEL__ >= 3 ) ) ) )
        = []( T_Neighbors const & x, T_Id const & i, size_t const & j ){ return x[i][j]; }
    #endif
    */
);


#include <LeMonADEGPU/utility/graphColoring.tpp>