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

#ifndef LEMONADEGPU_CORE_KERNELCONNECTION_H
#define LEMONADEGPU_CORE_KERNELCONNECTION_H
#include <stdint.h>
#include <LeMonADEGPU/utility/cudacommon.hpp>
class Connection
{
public: 
  Connection():arraySize(0), nThreads(1024){};
  Connection(uint32_t nReactiveMonomers_):arraySize(4*ceil((nReactiveMonomers_+1)*1.0/4.)), nThreads(1024), dBuffer(NULL){init();};
  
//   ~Connection(){clean();}
  ~Connection();
  
  void setNThreads(uint32_t nThreads_ ) {nThreads=nThreads_;}
  uint32_t getNThreads() const {return nThreads;}
//   void setArraySize(uint32_t arraySize_){arraySize=arraySize_;}
  uint32_t getArraySize() const {return arraySize;}
  void init();
  void clean();
  inline void setArraySize(uint32_t nReactiveMonomers_ ){arraySize=(4*ceil((nReactiveMonomers_+1)*1.0/4.));}
  /**
   * @brief resets the mutliple occurent ides to zero.
   * @details  we have to differentiate between two cases because the shared memory 
   * is restricted to 48 kB. For each entry in the array we need 2 times 4 byte(=uint32_t).
   * 48kb/8 ~ 6000. For higher number of cross links we need to use the global memory 
   * @todo optimize this function:
   * 	- reduce array size by remoiving (leading) zeros  after resorting 
   * 	- use again shared memory...	
   */
  void resetMultipleIDs( uint32_t * crosslinkId, uint32_t * chainID, cudaStream_t mStream );
  /**
   * @brief check if the connection partner in flags has found also a partner... 
   */
  void resetMultipleBonds( uint32_t * IDs, uint32_t * flags, cudaStream_t mStream );
  
private: 
  uint32_t nThreads;
  uint32_t arraySize;
  uint32_t * dBuffer;
};
#endif /*LEMONADEGPU_CORE_KERNELCONNECTION_H*/