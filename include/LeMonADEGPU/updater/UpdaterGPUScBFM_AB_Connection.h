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
#ifndef LEMONADEGPU_UPDATER_GPUCONNECTION_H_

#define LEMONADEGPU_UPDATER_GPUCONNECTION_H_

#include <LeMonADEGPU/updater/UpdaterGPUScBFM.h>
#include <LeMonADEGPU/utility/SelectiveLogger.hpp>
#include <LeMonADEGPU/utility/cudacommon.hpp>
#include <LeMonADEGPU/core/constants.cuh>
#include <LeMonADEGPU/utility/GPUConnectionTracker.h>
#include <LeMonADEGPU/core/kernelConnection.h>
struct D_MonomerReactivity {
    using T_Id               = UpdaterGPUScBFM< uint8_t >::T_Id         ;
    typedef uint8_t T_MaxNumLinks;
    T_Id monID;
    bool reactivity;
    T_MaxNumLinks maxNumLinks;
};


template< typename T_UCoordinateCuda >
class UpdaterGPUScBFM_AB_Connection: public UpdaterGPUScBFM<T_UCoordinateCuda>
{

public:
    typedef UpdaterGPUScBFM< T_UCoordinateCuda> BaseClass;
    using T_Flags            = UpdaterGPUScBFM<  uint8_t > :: T_Flags      ;
    using T_Lattice          = UpdaterGPUScBFM< uint8_t >::T_Lattice    ;
    using T_Coordinate       = UpdaterGPUScBFM< uint8_t >::T_Coordinate ;
    using T_Coordinates      = UpdaterGPUScBFM< uint8_t >::T_Coordinates;
    using T_Id               = UpdaterGPUScBFM< uint8_t >::T_Id         ;
    
    
    using T_MaxNumLinks = D_MonomerReactivity::T_MaxNumLinks;
    typedef uint32_t T_ReactiveLattice;
    using BaseClass::mLog;

protected:
    using BaseClass::mBoxX;
    using BaseClass::mBoxY;
    using BaseClass::mBoxZ;
    using BaseClass::mBoxXM1;
    using BaseClass::mBoxYM1;
    using BaseClass::mBoxZM1;
    using BaseClass::mBoxXLog2;
    using BaseClass::mBoxXYLog2;
    using BaseClass::met;
    using BaseClass::mStream;
    using BaseClass::mPolymerSystem;
    using BaseClass::mnAllMonomers;
    using BaseClass::mNeighbors;
    using BaseClass::checkBondVector;
    using BaseClass::mviPolymerSystemSortedVirtualBox;
    using BaseClass::mPolymerSystemSortedOld;
    using BaseClass::mPolymerSystemSorted;
    using BaseClass::mnElementsInGroup;
    using BaseClass::mCudaProps;
    using BaseClass::mAge;
    using BaseClass::useOverflowChecks;
    using BaseClass::findAndRemoveOverflows;
    using BaseClass::mnLatticeTmpBuffers;
    using BaseClass::mLatticeTmp;
    using BaseClass::mvtLatticeTmp;
    using BaseClass::randomNumbers;
    using BaseClass::launch_PerformSpeciesAndApply;
    using BaseClass::launch_PerformSpecies;
    using BaseClass::launch_ZeroArraySpecies;
    using BaseClass::miNewToi;
    using BaseClass::miToiNew;
    using BaseClass::mviSubGroupOffsets;
    using BaseClass::mNeighborsSorted;
    using BaseClass::mNeighborsSortedInfo;
    using BaseClass::mGroupIds;
    using BaseClass::mNeighborsSortedSizes;
    using BaseClass::hGlobalIterator;
    using BaseClass::doCopyBackMonomerPositions;
    using BaseClass::doCopyBackConnectivity;
    using BaseClass::diagMovesOn;

    uint32_t CrossLinkSpecies; 
    uint32_t ChainEndSpecies ; 
    size_t nReactiveMonomers;
    size_t nReactiveMonomersCrossLinks;
    size_t nReactiveMonomersChains;
    Tracker<T_UCoordinateCuda> tracker;
    Connection connection;
    
public:
    UpdaterGPUScBFM_AB_Connection();
    ~UpdaterGPUScBFM_AB_Connection();
private:
  
    /**
     * holds the reactivity and the
     * @todo I doubt that this must be a mirroredvector...
     * 	     it is used only on host.
     */
    std::vector< D_MonomerReactivity > mMonomerReactivity;
    //holds the IDS of the chains found by the conenction move 
    T_Id * mCrossLinkFlags;
    T_Id * mCrossLinkIDS;
    //must be a multiple of 4. This is a condition due to the used shared memory... 
    uint32_t flagArraySize; 
    //create a lattice with the ids on the edges
    MirroredTexture< T_Id > * mLatticeIds;
    
    std::vector< T_Id >  mNewToOldReactiveID;
    uint32_t crosslinkFunctionality, chainLength, nChains;

public:
    void initialize();
    void runSimulationOnGPU(const uint32_t nSteps );
    void doCopyBack();
    void checkSystem() const  ;
    void checkBonds() const ;
    void checkReactiveLatticeOccupation() ; 
    void cleanup();
    void destruct();

    void setNrOfReactiveMonomers ( T_Id nReactiveMonomers_, T_Id nReactiveMonomersCrossLinks_, T_Id nReactiveMonomersChains_ );
    void setChainLength(uint32_t  const chainLength_ );
    void setNChains(uint32_t  const nChains_ );
    
    void setReactiveGroup ( T_Id monID_, bool reactivity_, T_MaxNumLinks maxNumLinks_ );
    void initializeReactiveLattice();
    void launch_CheckConnection(
	  const size_t nBlocks, const size_t nThreads, const size_t iSpecies, 
	  const size_t iOffsetLatticeTmp, const uint64_t seed);
    void launch_initializeReactiveLattice(
	  const size_t nBlocks , const size_t nThreads, const T_Id iSpecies );
    void launch_resetReactiveLattice(
	  const size_t nBlocks , const size_t nThreads, const T_Id iSpecies );
    void launch_ApplyConnection(
	  const size_t nBlocks , const size_t   nThreads, 
	  const size_t MonomerSpecies,
	  const size_t PartnerSpecies);
};
#endif
