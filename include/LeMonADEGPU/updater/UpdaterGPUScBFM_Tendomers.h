#ifndef LEMONADEGPU_UPDATER_GPUCONNECTION_H_

#define LEMONADEGPU_UPDATER_GPUCONNECTION_H_

#include <LeMonADEGPU/updater/UpdaterGPUScBFM_AB_Type.h>
#include <LeMonADEGPU/utility/SelectiveLogger.hpp>
#include <LeMonADEGPU/utility/cudacommon.hpp>
#include <LeMonADEGPU/core/constants.cuh>

template< typename T_UCoordinateCuda >
class UpdaterGPUScBFM_Tendomers: public UpdaterGPUScBFM_AB_Type<T_UCoordinateCuda>
{

public:
    typedef UpdaterGPUScBFM_AB_Type< T_UCoordinateCuda> BaseClass;
    using T_Flags            = UpdaterGPUScBFM_AB_Type<  uint8_t > :: T_Flags   ;
    using T_Lattice          = UpdaterGPUScBFM_AB_Type< uint8_t >::T_Lattice    ;
    using T_Coordinate       = UpdaterGPUScBFM_AB_Type< uint8_t >::T_Coordinate ;
    using T_Coordinates      = UpdaterGPUScBFM_AB_Type< uint8_t >::T_Coordinates;
    using T_Id               = UpdaterGPUScBFM_AB_Type< uint8_t >::T_Id         ;
    using T_Color            = UpdaterGPUScBFM_AB_Type< uint8_t >::T_Color      ;
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
    using BaseClass::mPolymerSystemSortedOld;
    using BaseClass::mPolymerSystemSorted;
    using BaseClass::mnElementsInGroup;
    using BaseClass::mCudaProps;
    using BaseClass::mAge;
    using BaseClass::mUsePeriodicMonomerSorting;
    using BaseClass::mnStepsBetweenSortings;
    using BaseClass::doSpatialSorting;
    using BaseClass::useOverflowChecks;
    using BaseClass::findAndRemoveOverflows;
    using BaseClass::mnLatticeTmpBuffers;
    using BaseClass::mLatticeTmp;
    using BaseClass::mvtLatticeTmp;
    using BaseClass::randomNumbers;
    using BaseClass::launch_CheckSpecies;
    using BaseClass::launch_CheckReactiveSpecies;
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
    using BaseClass::mGlobalIterator;
    using BaseClass::doCopyBackMonomerPositions;
    using BaseClass::doCopyBackConnectivity;	
    using BaseClass::mPolymerFlags;
    using BaseClass::mLatticeOut;
    using BaseClass::boxCheck;

//     //flag to decide which monomer of the AA bond pair should move in which subsubstep
//     MirroredTexture< uint8_t > * AAMonomerFlag;
    
public:
    UpdaterGPUScBFM_Tendomers();
    ~UpdaterGPUScBFM_Tendomers();
private:
  
    //create a lattice with the ids on the edges
    MirroredTexture< T_Id > * mLatticeIds;
    uint32_t nSegmentsPerChain;
    uint32_t nTendomers;

public:
  
    void set
    void initialize();
    void runSimulationOnGPU(const uint32_t nSteps );
    void doCopyBack();
    void checkSystem() const  ;
    void checkBonds() const ;
    void cleanup();
    void destruct();
    

};
#endif