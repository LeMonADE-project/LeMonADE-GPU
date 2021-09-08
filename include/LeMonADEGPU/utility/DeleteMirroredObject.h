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
/**
 * @brief convinience class for a fast deletation of objects....
 */
struct DeleteMirroredObject
{
    size_t nBytesFreed = 0;

    template< typename S >
    void operator()( MirroredVector< S > * & p, std::string const & name = "" )
    {
        if ( p != NULL )
        {
            std::cerr
                << "Free MirroredVector " << name << " at " << (void*) p
                << " which holds " << prettyPrintBytes( p->nBytes ) << "\n";
            nBytesFreed += p->nBytes;
            delete p;
            p = NULL;
        }
    }

    template< typename S >
    void operator()( MirroredTexture< S > * & p, std::string const & name = "" )
    {
        if ( p != NULL )
        {
            nBytesFreed += p->nBytes;
            delete p;
            p = NULL;
        }
    }
};