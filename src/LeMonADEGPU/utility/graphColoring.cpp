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
#include <LeMonADEGPU/utility/graphColoring.h>
// #include <LeMonADEGPU/utility/graphColoring.tpp>

#include <vector>


#define INSTANTIATE_TMP( T_Neighbors, T_Id, T_Color )                                                \
template std::vector< T_Color > graphColoring< T_Neighbors >                                         \
(                                                                                                    \
    T_Neighbors const & rvNeighbors,                                                                 \
    size_t      const & rnElements ,                                                                 \
    bool        const   rbUniformColors,                                                             \
    std::function< size_t( T_Neighbors const &, T_Id const & ) > const & rfGetNeighborsSize,         \
    std::function< T_Id( T_Neighbors const &, T_Id const &, size_t const & ) > const & rfGetNeighbor \
);
/* vector / list for each monomer */
INSTANTIATE_TMP( std::vector< std::vector< uint8_t > >, uint8_t , uint8_t  )
INSTANTIATE_TMP( std::vector< std::vector< uint8_t > >, uint8_t , uint16_t )
INSTANTIATE_TMP( std::vector< std::vector< uint8_t > >, uint8_t , uint32_t )
INSTANTIATE_TMP( std::vector< std::vector< uint8_t > >, uint16_t, uint8_t  )
INSTANTIATE_TMP( std::vector< std::vector< uint8_t > >, uint16_t, uint16_t )
INSTANTIATE_TMP( std::vector< std::vector< uint8_t > >, uint16_t, uint32_t )
INSTANTIATE_TMP( std::vector< std::vector< uint8_t > >, uint32_t, uint8_t  )
INSTANTIATE_TMP( std::vector< std::vector< uint8_t > >, uint32_t, uint16_t )
INSTANTIATE_TMP( std::vector< std::vector< uint8_t > >, uint32_t, uint32_t )

#undef INSTANTIATE_TMP

