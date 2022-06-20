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
#ifndef LEMONADEGPU_UPDATER_GPUREAVERSIBLECONNECTION_H_
#define LEMONADEGPU_UPDATER_GPUREAVERSIBLECONNECTION_H_

#include <LeMonADEGPU/updater/UpdaterGPUScBFM_AB_Connection.h>
#include <LeMonADEGPU/utility/SelectiveLogger.hpp>
#include <LeMonADEGPU/utility/cudacommon.hpp>
#include <LeMonADEGPU/core/constants.cuh>
#include <LeMonADEGPU/utility/GPUConnectionTracker.h>

template< typename T_UCoordinateCuda >
class UpdaterGPUScBFM_AB_Breaking: public UpdaterGPUScBFM_AB_Connection<T_UCoordinateCuda>
{

public:
    typedef UpdaterGPUScBFM_AB_Connection< T_UCoordinateCuda> BaseClass;
    using T_Flags            = UpdaterGPUScBFM_AB_Connection<  uint8_t > :: T_Flags      ;
    using T_Lattice          = UpdaterGPUScBFM_AB_Connection< uint8_t >::T_Lattice    ;
    using T_Coordinate       = UpdaterGPUScBFM_AB_Connection< uint8_t >::T_Coordinate ;
    using T_Coordinates      = UpdaterGPUScBFM_AB_Connection< uint8_t >::T_Coordinates;
    using T_Id               = UpdaterGPUScBFM_AB_Connection< uint8_t >::T_Id         ;
    using BaseClass::mLog;

protected:
    using BaseClass::mBoxX;
    using BaseClass::mBoxY;
    using BaseClass::mBoxZ;
    using BaseClass::met;
    using BaseClass::mStream;
    using BaseClass::mnAllMonomers;
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
    using BaseClass::mviSubGroupOffsets;
    using BaseClass::mNeighborsSorted;
    using BaseClass::mNeighborsSortedInfo;
    using BaseClass::mNeighborsSortedSizes;
    using BaseClass::hGlobalIterator;
    using BaseClass::doCopyBackMonomerPositions;
    using BaseClass::doCopyBackConnectivity;
    using BaseClass::launch_CheckConnection;
    using BaseClass::launch_initializeReactiveLattice; 
    using BaseClass::launch_resetReactiveLattice;
    using BaseClass::launch_ApplyConnection;
    using BaseClass::checkReactiveLatticeOccupation;
    using BaseClass::nReactiveMonomersCrossLinks;
    using BaseClass::CrossLinkSpecies;
    using BaseClass::ChainEndSpecies;
    using BaseClass::tracker;
    using BaseClass::diagMovesOn;
    
public:
    UpdaterGPUScBFM_AB_Breaking();
    ~UpdaterGPUScBFM_AB_Breaking();
private:
    double energy;
    MirroredVector<T_Id> * dBreaksID1;
    MirroredVector<T_Id> * dBreaksID2;
public:
    void setBondEnergy(double energy_);
    void initialize();
    void runSimulationOnGPU(const uint32_t nSteps );
    void doCopyBack();
    void checkSystem() const  ;
    void cleanup();
    void destruct();

    void launch_BreakConnections(
	  const size_t nBlocks, const size_t nThreads, const size_t iSpecies, 
	  const size_t iOffsetLatticeTmp, const uint64_t seed);
};
#endif