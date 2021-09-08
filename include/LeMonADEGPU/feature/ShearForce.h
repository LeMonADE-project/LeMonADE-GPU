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
struct ShearForce
{
// doing the Metropolis-criterion for movement
// dU = gamma*dr
// slice z=0 && z=1 -> right gamma=(1,0,0) -> dU = dx
// slice z is [Box/2-3;Box/2+1] -> left gamma=(-1,0,0) -> dU = -dx
// slice z=Box-1 && z=Box-2 -> right gamma=(1,0,0) -> dU = dx
// otherwise -> always allowed  
//   ShearForce(){}
  template <class T1, class T2>
  inline __device__ double  operator() (T1 ZPosition, T2 dx)
  {
    uint32_t id(0);
    if      ( ZPosition < 2                                              ) id += 1;
    else if ((ZPosition <= dcBoxZ_HalfP1) && (ZPosition > dcBoxZ_HalfM3) ) id += 2;
    else if ( ZPosition >= dcBoxZM2                                      ) id += 3;
    else return 1;
    //  Metropolis-criterion is only necessary
    if ( dx < 0 )
      id+=3;
    return dLookUpShearForce[id | 2];

  }
};