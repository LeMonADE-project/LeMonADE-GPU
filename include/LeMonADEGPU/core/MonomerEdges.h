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
/**
 * When not reordering the neighbor information as struct of array,
 * then increasing this leads to performance degradataion!
 * But currently, as the reordering is implemented, it just leads to
 * higher memory usage.
 * In the 3D case more than 20 makes no sense for the standard bond vector
 * set, as the volume exclusion plus the bond vector set make 20 neighbors
 * the maximum. In real use cases 8 are already very much / more than sufficient.
 */
#pragma once
#define MAX_CONNECTIVITY 8
/* stores amount and IDs of neighbors for each monomer */
struct MonomerEdges
{
    uint32_t size; // could also be uint8_t as it is limited by MAX_CONNECTIVITY
    uint32_t neighborIds[ MAX_CONNECTIVITY ];
};
