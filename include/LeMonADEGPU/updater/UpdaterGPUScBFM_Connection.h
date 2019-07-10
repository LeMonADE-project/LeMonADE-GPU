#ifndef LEMONADEGPU_UPDATER_GPUCONNECTION_H_

#define LEMONADEGPU_UPDATER_GPUCONNECTION_H_

#include <LeMonADEGPU/updater/UpdaterGPUScBFM_AB_Type.h>
#include <LeMonADEGPU/utility/SelectiveLogger.hpp>
#include <LeMonADEGPU/utility/cudacommon.hpp>
#include <LeMonADEGPU/core/constants.cuh>
struct D_MonomerReactivity {
    typedef uint8_t T_MaxNumLinks;
    T_Id monID;
    bool reactivity;
    T_MaxNumLinks maxNumLinks;
};


template< typename T_UCoordinateCuda >
class UpdaterGPUScBFM_Connection: public UpdaterGPUScBFM_AB_Type<T_UCoordinateCuda>
{

public:
    typedef UpdaterGPUScBFM_AB_Type< T_UCoordinateCuda> BaseClass;
    using T_Flags            = UpdaterGPUScBFM_AB_Type<  uint8_t > :: T_Flags      ;
    using T_Lattice          = UpdaterGPUScBFM_AB_Type< uint8_t >::T_Lattice    ;
    using T_Coordinate       = UpdaterGPUScBFM_AB_Type< uint8_t >::T_Coordinate ;
    using T_Coordinates      = UpdaterGPUScBFM_AB_Type< uint8_t >::T_Coordinates;
    using T_Id               = UpdaterGPUScBFM_AB_Type< uint8_t >::T_Id         ;
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
    using BaseClass::launch_CheckSpecies;
    using BaseClass::launch_PerformSpeciesAndApply;
    using BaseClass::launch_PerformSpecies;
    using BaseClass::launch_ZeroArraySpecies;
    using BaseClass::miNewToi;
    using BaseClass::mviSubGroupOffsets;
    using BaseClass::mNeighborsSorted;
    using BaseClass::mNeighborsSortedInfo;
    using BaseClass::mGroupIds;
    using BaseClass::mNeighborsSortedSizes;
    using BaseClass::mGlobalIterator;
    using BaseClass::doCopyBackMonomerPositions;
    using BaseClass::doCopyBackConnectivity;

    void doCopyBack();
    void launch_CheckConnection(const size_t nBlocks, const size_t nThreads, const size_t iSpecies, const size_t iOffsetLatticeTmp, const uint64_t seed);
    
public:
    UpdaterGPUScBFM_Connection();
    ~UpdaterGPUScBFM_Connection();
private:
  
    /**
     * holds the reactivity and the
     * @todo I doubt that this must be a mirroredvector...
     * 	     it is used only on host.
     */
//     MirroredVector< D_MonomerReactivity > * mMonomerReactivity;
    std::vector< D_MonomerReactivity > mMonomerReactivity;
    //holds the IDS of the chains found by the conenction move 
//     MirroredVector< T_Id > * mCrossLinkFlags;
//     MirroredVector< T_Id > * mCrossLinkIDS;
    T_Id * mCrossLinkFlags;
    T_Id * mCrossLinkIDS;
    //must be a multiple of 4. This is a condition due to the used shared memory... 
    uint32_t flagArraySize; 
    uint32_t CrossLinkSpecies; 
    uint32_t ChainEndSpecies ; 
    //create a lattice with the ids on the edges
    MirroredTexture< T_Id > * mLatticeIds;
    
//     MirroredVector< T_Id > * mNewToOldReactiveID;
    std::vector< T_Id >  mNewToOldReactiveID;
    uint32_t crosslinkFunctionality;
//   void  getCompressesReactivity(D_MonomerReactivity& monReact, T_ReactiveLattice& entry);
//     T_ReactiveLattice  getCompressesReactivity ( D_MonomerReactivity& monReact ) ;
//     D_MonomerReactivity setCompressesReactivity ( T_Id monID_, bool reactivity_, T_MaxNumLinks maxNumLinks_ );
    

    size_t nReactiveMonomers;
    size_t nReactiveMonomersCrossLinks;
    size_t nReactiveMonomersChains;
public:
    void initialize();
//     bool execute();
    void runSimulationOnGPU(const uint32_t nSteps );
//     void doCopyBack();
    void checkSystem() const  ;
    void checkBonds() const ;
    void checkLatticeOccupation() ; 
    void cleanup();
    void destruct();
    
    
    
    void setNrOfReactiveMonomers ( T_Id nReactiveMonomers_, T_Id nReactiveMonomersCrossLinks_, T_Id nReactiveMonomersChains_ );
    void setReactiveGroup ( T_Id monID_, bool reactivity_, T_MaxNumLinks maxNumLinks_ );
    void initializeReactiveLattice();
    void launch_initializeReactiveLattice(
	  const size_t nBlocks , const size_t nThreads, const T_Id iSpecies );
    void launch_CheckConnection(
	  const size_t nBlocks, const size_t nThreads, 
	  const size_t iSpecies, const uint64_t seed);
    void launch_ApplyConnection(
	  const size_t nBlocks , const size_t   nThreads, 
	  const size_t MonomerSpecies,
	  const size_t PartnerSpecies);
};
#endif