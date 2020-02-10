#ifndef LEMONADEGPU_UPDATER_GPUCONNECTION_H_

#define LEMONADEGPU_UPDATER_GPUCONNECTION_H_

#include <LeMonADEGPU/updater/UpdaterGPUScBFM.h>
#include <LeMonADEGPU/utility/SelectiveLogger.hpp>
#include <LeMonADEGPU/utility/cudacommon.hpp>
#include <LeMonADEGPU/core/constants.cuh>
#include <LeMonADEGPU/utility/GPUConnectionTracker.h>
#include <LeMonADEGPU/core/kernelConnection.h>
struct D_MonomerReactivity {
    typedef uint8_t T_MaxNumLinks;
    T_Id monID;
    bool reactivity;
    T_MaxNumLinks maxNumLinks;
};


template< typename T_UCoordinateCuda >
class UpdaterGPUScBFM_AA_Connection: public UpdaterGPUScBFM<T_UCoordinateCuda>
{

public:
    typedef UpdaterGPUScBFM< T_UCoordinateCuda> BaseClass;
    using T_Flags            = UpdaterGPUScBFM<  uint8_t > :: T_Flags   ;
    using T_Lattice          = UpdaterGPUScBFM< uint8_t >::T_Lattice    ;
    using T_Coordinate       = UpdaterGPUScBFM< uint8_t >::T_Coordinate ;
    using T_Coordinates      = UpdaterGPUScBFM< uint8_t >::T_Coordinates;
    using T_Id               = UpdaterGPUScBFM< uint8_t >::T_Id         ;
    using T_Color            = UpdaterGPUScBFM< uint8_t >::T_Color      ;
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
    using BaseClass::mPolymerFlags;
    using BaseClass::mLatticeOut;
    using BaseClass::boxCheck;
    using BaseClass::diagMovesOn;

    uint32_t ChainEndSpecies ; 
    size_t nReactiveMonomers;
    Tracker tracker;
    Connection connection;
    //flag to decide which monomer of the AA bond pair should move in which subsubstep
    MirroredTexture< uint8_t > * AAMonomerFlag;
    
public:
    UpdaterGPUScBFM_AA_Connection();
    ~UpdaterGPUScBFM_AA_Connection();
private:
  
    /**
     * holds the reactivity and the
     * @todo I doubt that this must be a mirroredvector...
     * 	     it is used only on host.
     */
    std::vector< D_MonomerReactivity > mMonomerReactivity;
    //holds the IDS of the chains found by the conenction move 
    T_Id * mChainEndFlags;
    T_Id * mChainEndIDS;
    //must be a multiple of 4. This is a condition due to the used shared memory... 
    uint32_t flagArraySize; 
    //create a lattice with the ids on the edges
    MirroredTexture< T_Id > * mLatticeIds;
    
    std::vector< T_Id >  mNewToOldReactiveID;
    uint32_t crosslinkFunctionality;

public:
    void initialize();
    void runSimulationOnGPU(const uint32_t nSteps );
    void doCopyBack();
    void checkSystem() const  ;
    void checkBonds() const ;
    void checkReactiveLatticeOccupation() ; 
    void cleanup();
    void destruct();

    void setNrOfReactiveMonomers ( T_Id nReactiveMonomers_);
    void setReactiveGroup ( T_Id monID_, bool reactivity_, T_MaxNumLinks maxNumLinks_ );
    void initializeReactiveLattice();
    
    void launch_CheckConnection(
	  const size_t nBlocks , const size_t nThreads, 
	  const size_t iSpecies, const uint64_t seed);
    void launch_initializeReactiveLattice(
	  const size_t nBlocks , const size_t nThreads, const T_Id iSpecies );
    void launch_resetReactiveLattice(
	  const size_t nBlocks , const size_t nThreads, const T_Id iSpecies );
    void launch_ApplyConnection(
	  const size_t nBlocks , const size_t   nThreads, 
	  const size_t MonomerSpecies);
};
#endif