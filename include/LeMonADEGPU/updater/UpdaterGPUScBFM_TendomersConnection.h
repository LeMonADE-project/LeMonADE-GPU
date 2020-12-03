#ifndef LEMONADEGPU_UPDATER_GPUTENDOMERCONNECTION_H_

#define LEMONADEGPU_UPDATER_GPUTENDOMERCONNECTION_H_

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
class UpdaterGPUScBFM_TendomersConnection: public UpdaterGPUScBFM<T_UCoordinateCuda>
{

public:
    typedef UpdaterGPUScBFM< T_UCoordinateCuda> BaseClass;
    using T_Flags            = UpdaterGPUScBFM<  uint8_t > :: T_Flags   ;
    using T_Lattice          = UpdaterGPUScBFM< uint8_t >::T_Lattice    ;
    using T_Coordinate       = UpdaterGPUScBFM< uint8_t >::T_Coordinate ;
    using T_Coordinates      = UpdaterGPUScBFM< uint8_t >::T_Coordinates;
    using T_Id               = UpdaterGPUScBFM< uint8_t >::T_Id         ;
    using T_Color            = UpdaterGPUScBFM< uint8_t >::T_Color      ;
    using T_Label            = uint8_t                                  ;
    using T_RingCoordinates  = int2; // this cannot be used on the CPU,or ? 
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
   
    uint32_t CrossLinkSpecies; 
    uint32_t ChainEndSpecies ; 
    size_t nReactiveMonomers;
    size_t nReactiveMonomersCrossLinks;
    size_t nReactiveMonomersChains;
    Tracker<T_UCoordinateCuda> tracker;
    Connection connection;

public:
    UpdaterGPUScBFM_TendomersConnection();
    ~UpdaterGPUScBFM_TendomersConnection();
private:
    ///////////////////////start connection part ////////////////
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
    uint32_t crosslinkFunctionality;
    ///////////////////////end connection part ////////////////
    ///////////////////////start label part    ////////////////
    uint32_t nMonomersPerChain, nTendomers, nCrossLinks, nLabelsPerTendomerArm, functionality, nLabels;
    //!
    std::vector< uint32_t > labelOffset;
    //!number of labels per species 
    std::vector< uint32_t > nLabelsPerSpecies;
    //!maybe this could be a normal std::vector....
    std::vector<uint32_t> vMonomerLabel;
    //!ring id is the vector index and the value is the label ( usually for the tendomer its 6 and 7)
    std::vector<uint32_t> vLabelValue;
    
    // stores if it is occupied by a label and the ID of the corresponding chain monomers 
    MirroredVector< T_Id > * mLatticeLabel; 
    // https://stackoverflow.com/questions/19777910/how-make-int2-is-working
    MirroredVector< T_RingCoordinates > * mLabelPosition;
    //connectivity of the slide ring 
    MirroredVector< T_RingCoordinates > * mLabelBonds;
    //can be either 0:standard moves for all monomers, 1: diagonal moves for all monomers, 2: diagonal moves for pending chain and standard moves for elastic chain monomers
    int monomericMoveType;
    //stores if the monomer can use diagonal moves or not
    MirroredTexture<  uint8_t > * moveType;
    ///////////////////////end label part    //////////////////
public:
  
    //setter functions 
    void    setNTendomers             ( uint32_t nTendomers_            ); 
    void    setNumCrossLinkers        ( uint32_t nCrossLinks_           );
    void    setNumMonomersPerChain    ( uint32_t nMonomersPerChain_     );
    void    setNumLabelsPerTendomerArm( uint32_t nLabelsPerTendomerArm_ );
    void    setFunctionality          ( uint32_t functionality_         );
    void    setLabel                  ( uint32_t ID_, uint32_t label_   );
    void    setMoveType               ( int      monomericMoveType_     );
    
    int32_t getLabel                  ( uint32_t ID_ );
    
    void initialize();
    void runSimulationOnGPU(const uint32_t nSteps );
    void launch_MoveLabel(const size_t nBlocks, const size_t nThreads, const size_t iSpecies, const uint64_t seed);
    void doCopyBack();
    void doCopyBackLabels();
    void checkSystem() const  ;
    void checkBonds() const ;
    void cleanup();
    void destruct();
    

    void checkReactiveLatticeOccupation() ; 
    void setNrOfReactiveMonomers ( T_Id nReactiveMonomers_, T_Id nReactiveMonomersCrossLinks_, T_Id nReactiveMonomersChains_ );
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