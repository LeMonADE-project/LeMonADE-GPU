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
#ifndef LEMONADEGPU_CORE_METHOD_H_
#define LEMONADEGPU_CORE_METHOD_H_
#include <LeMonADEGPU/core/SpaceFillingCurve.h>
#include <LeMonADEGPU/core/BitPacking.h>

/**
 * @brief convinience class storing methods which are used in the kernel 
 */
class Method {
 
 private:    
     SpaceFillingCurve curve; 
  
 public:
    __device__ __host__ const SpaceFillingCurve&   getCurve() const 
    {return curve; }

   __device__ __host__ SpaceFillingCurve&   modifyCurve() 
   {return curve; }
   
  private:
      BitPacking packing;
  public:
    __device__ __host__ const BitPacking& getPacking() const 
    {return packing; }
    __device__ __host__ BitPacking& modifyPacking() 
    {return packing;}
  private:
    bool useGPUForOverhead;
  public:
    bool isONGPUForOverhead() const 
    {return useGPUForOverhead;}
    void setOnGPUForOverhead(const bool useGPUForOverhead_)
    {useGPUForOverhead=useGPUForOverhead_;}
  
};
#endif /*LEMONADEGPU_CORE_METHOD_H_*/