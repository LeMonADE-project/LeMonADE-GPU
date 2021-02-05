/*
 * UpdaterGPUScBFM.h
 *
 *  Created on: 27.07.2017
 *      Authors: Ron Dockhorn, Maximilian Knespel
 */

#pragma once


#include <cassert>
#include <chrono>                           // high_resolution_clock
#include <cstdio>                           // printf
#include <cstdint>                          // uint32_t, size_t
#include <stdexcept>
#include <type_traits>                      // make_unsigned

#include <cuda_runtime_api.h>               // cudaStream_t, cudaDeviceProp
#include <curand.h>

#include <LeMonADE/utility/RandomNumberGenerators.h>
#include <LeMonADEGPU/utility/cudacommon.hpp>
#include <LeMonADEGPU/utility/SelectiveLogger.hpp>
#include <LeMonADEGPU/core/rngs/Saru.h>
#include <LeMonADEGPU/core/constants.cuh>
#include <LeMonADEGPU/utility/alignedMatrizes.h>
#include <LeMonADEGPU/core/MonomerEdges.h>
#include <LeMonADEGPU/core/BondVectorSet.h>
#include <LeMonADEGPU/core/Method.h>
#include <LeMonADEGPU/feature/BoxCheck.h>

//keep this 
// #define USE_BIT_PACKING_TMP_LATTICE
// #define USE_ZCURVE_FOR_LATTICE
// #if defined( USE_BIT_PACKING_TMP_LATTICE )
// #   define USE_NBUFFERED_TMP_LATTICE
// #endif
// #define USE_GPU_FOR_OVERHEAD

//erase that
// #define USE_BIT_PACKING_LATTICE


/**
 * working combinations:
 *   - nothing set (3.96109s)
 *   - USE_ZCURVE_FOR_LATTICE (2.18166s)
 *   - USE_BIT_PACKING_TMP_LATTICE + USE_ZCURVE_FOR_LATTICE (1.56671s)
 * not working:
 *   - USE_BIT_PACKING_TMP_LATTICE
 * @todo using USE_BIT_PACKING_TMP_LATTICE without USE_ZCURVE_FOR_LATTICE
 *       does not work! The error must be in checkFront ... ?
 */

template< typename T_UCoordinateCuda > 
class UpdaterGPUScBFM
{

public:

    /**
     * This is still used, at least the 32 bit version!
     * These are the types for the monomer positions.
     * Even for the periodic case, we are sometimes interested in the "global"
     * position without fold back to the periodic box in order to calculate
     * diffusion easily.
     * Although you also could derive an algorithm for automatically finding
     * larger jumps and undo the folding back manually. In order to not miss
     * a case you would have to do this every
     *     dtApplyJumps = ceilDiv( min( boxX, boxY, boxZ ), 2 ) - 1
     * time steps. If a particle moved dtApplyJumps+1, then you could be sure
     * that it was indeed because of the periodicity condition.
     * But then again, using uint8_t would limit the box size to 256 which we
     * do not want. The bit operation introduced limitation to 1024 is already
     * a little bit worrisome.
     * But IF you have such a small box and you can use uint8_t, then we could
     * do as written above and possibly make the algorithm EVEN FASTER as the
     * memory bandwidth could be reduced even more!
     */
    //type for size of the simulation box
    using T_BoxSize          = uint64_t; // uint32_t // should be unsigned!
    
    //type for coordinate for the host 
    using T_Coordinate       = int32_t; // int64_t // should be signed!
    //type for vector of coordinates on host 
    using T_Coordinates      = typename CudaVec4< T_Coordinate      >::value_type;
    
    //type for unsigned coordinates on the device 
    using T_UCoordinatesCuda = typename CudaVec4< T_UCoordinateCuda >::value_type;

    /* could also be uint8_t if you know you only have 256 different
     * species at maximum. For the autocoloring this is implicitly true,
     * but not so if the user manually specifies colors! */
    using T_Color   = uint32_t;
    //flag to store the result of an attempted move 
    using T_Flags   = uint8_t ; // uint16_t, uint32_t
    //mostly used for the monomer Ids 
    using T_Id      = uint32_t; // should be unsigned!
    //lattice entry 
    using T_Lattice = uint8_t ; // untested for something else than uint8_t!
    
    // getBitPackedTextureFunction is a pointer to a function which takes (cudaTextureObject_t tex, int i) as arguments and return T_Lattice
    typedef  T_Lattice (BitPacking::*getBitPackedTextureFunction)(cudaTextureObject_t tex, int i) const ;  

    /** @brief tracks the output for certain streams, like: 'stat', 'check', ...
     */
    SelectedLogger mLog;

protected:
    /* only do checks for uint8_t and uint16_t, for all signed data types
     * we kinda assume the user to think himself smart enough that he does
     * not want the overflow checking version. This is because the overflow
     * checking is not tested with signed types being used! 
     * Previously this was a 'static bool constexp' which is not useful
     * because the periodicity is only known at runtime and is one 
     * contribution of setting to true or false.
     */
    bool useOverflowChecks;

    /**
     * @brief up to now there is only the default stream used
     * @details this could be extended to more streams for concurrency running kernels 
     */
    cudaStream_t mStream;
    /**
     * Vector of length boxX * boxY * boxZ. Actually only contains 0 if
     * the cell / lattice point is empty or 1 if it is occupied by a monomer
     * Suggestion: bitpack it to save 8 times memory and possibly make the
     *             the reading faster if it is memory bound ???
     */

    MirroredTexture< T_Lattice > * mLatticeOut, * mLatticeTmp, * mLatticeTmp2;
    /**
     * when using bit packing only 1/8 of mLatticeTmp is used. In order to
     * to using everything we can simply increment the mLatticeTmp->gpu pointer,
     * but for textures this is not possible so easily, therefore we need to
     * store texture objects for each bit packed sub lattice in mLatticeTmp.
     * Then after 8 usages we can call one cudaMemset for all, possibly making
     * 8 times better use of parallelism on the GPU!
     */
//     static auto constexpr mnLatticeTmpBuffers = 2u; // brings a undefined reference for gcc 7.3, but not for gcc 4.85 ?!
    const uint mnLatticeTmpBuffers = 2u;

    std::vector< cudaTextureObject_t > mvtLatticeTmp;

    /* copy into mPolymerSystem and drop the property tag while doing so.
     * would be easier and probably more efficient if mPolymerSystem_device/host
     * would be a struct of arrays instead of an array of structs !!! */
    /**
     * Contains the nMonomers particles as well as a property tag for each:
     *   [ x0, y0, z0, p0, x1, y1, z1, p1, ... ]
     * The property tags p are bit packed:
     * @verbatim
     *                        8  7  6  5  4  3  2  1  0
     * +--------+--+--+--+--+--+--+--+--+--+--+--+--+--+
     * | unused |  |  |  |  |c |   nnr  |  dir   |move |
     * +--------+--+--+--+--+--+--+--+--+--+--+--+--+--+
     *  c   ... charged: 0 no, 1: yes
     *  nnr ... number of neighbors, this will get populated from LeMonADE's
     *          get get
     *  move ... Bit 0 set by kernelCheckSpecies if move was found to be possible
     *           Bit 1 set by kernelPerformSpecies if move still possible
     *           heeding the new locations of all particles.
     *           If both bits set, the move gets applied to polymerSystem in
     *           kernelZeroArrays
     * @endverbatim
     * The saved location is used as the lower left front corner when
     * populating the lattice with 2x2x2 boxes representing the monomers
     */

    size_t mnAllMonomers;
    MirroredVector< T_Coordinates > * mPolymerSystem;
    /**
     * This is mPolymerSystem sorted by species and also made struct of array
     * in order to split neighbors size off into extra array, thereby also
     * increasing max neighbor size from 8 to 256!
     * @verbatim
     * A1x A2x A3x A4x ... A1y A2y A3y A4y ... A1z A2z ... B1x B2x ...
     * @endverbatim
     * Note how this struct of array leads to yet another alignment problem
     * I think I need AlignedMatrices for this, too :(
     */
    size_t mnMonomersPadded;
    MirroredVector< T_UCoordinatesCuda > * mPolymerSystemSorted;
    MirroredVector< T_UCoordinatesCuda > * mPolymerSystemSortedOld;
    MirroredVector< T_Coordinates      > * mviPolymerSystemSortedVirtualBox;
    /**
     * These are to be used for storing the flags and chosen direction of
     * the old property tag.
     *      4  3  2  1  0
     *    +--+--+--+--+--+
     *    |  dir   |move |
     *    +--+--+--+--+--+
     * These are currently temporary vectors only written and read to from
     * the GPU, so MirroredVector isn't necessary, but it's easy to use and
     * could be nice for debugging (e.g. to replace the count kernels)
     */
    MirroredVector< T_Flags > * mPolymerFlags;

    static auto constexpr nBytesAlignment    = 512u;
    static auto constexpr nElementsAlignment = nBytesAlignment / ( 4u * sizeof( T_UCoordinateCuda ) );
    static_assert( nBytesAlignment == nElementsAlignment * 4u * sizeof( T_UCoordinateCuda ),
                   "Element type of polymer systems seems to be larger than the Bytes we need to align on!" );

    /* for each monomer the attribute 1 (A) or 2 (B) is stored
     * -> could be 1 bit per monomer ... !!!
     * @see http://www.gotw.ca/gotw/050.htm
     * -> wow std::vector<bool> already optimized for space with bit masking!
     * This is only needed once for initializing mMonomerIdsA,B */
    std::vector< int32_t > mAttributeSystem;
    std::vector< T_Color > mGroupIds; /* for each monomer stores the color / attribute / group ID/tag */
    std::vector< size_t  > mnElementsInGroup;
    std::vector< T_Id    > mviSubGroupOffsets; /* stores offsets (in number of elements not bytes) to each aligned subgroup vector in mPolymerSystemSorted */
    MirroredVector< T_Id > * miToiNew; /* for each old monomer stores the new position */
    MirroredVector< T_Id > * miNewToi; /* for each new monomer stores the old position */
    MirroredVector< T_Id > * miNewToiComposition; /* temporary buffer for storing result of miNewToi[ miNewToiSpatial ] */
    MirroredVector< T_Id > * miNewToiSpatial; /* used for sorting monomers along z-curve on GPU */
    MirroredVector< T_Id > * mvKeysZOrderLinearIds; /* used for sorting monomers along z-curve on GPU */

    /* set autocoloring or manual coloring*/
    bool bSetAutoColoring;
    /* in order to decide how to work with setMonomerCoordinates and
     * getMonomerCoordinates. This way we might just return the monomer
     * info directly instead of needing to desort after each execute run */
    bool bPolymersSorted;

    MirroredVector< MonomerEdges > * mNeighbors;
    /**
     * stores the IDs of all neighbors as is needed to check for the bond
     * set / length restrictions.
     * But after the sorting of mPolymerSystem the IDs also changed.
     * And as I don't want to push miToiNew and miNewToi to the GPU instead
     * I just change the IDs for all neighbors. Plus the array itself gets
     * rearranged to the new AAA...BBB...CC... ordering
     *
     * In the next level optimization this sorting, i.e. A{neighbor 1234}A...
     * Needs to get also resorted to:
     *
     * @verbatim
     *       nMonomersPaddedInGroup[0] = 8
     *    <---------------------------->
     *    A11 A21 A31 A41 A51  0   0   0
     *    A21 A22 A32 A42 A52  0   0   0
     *    ...
     *    nMonomersPaddedInGroup[1] = 4
     *    <------------>
     * +> B11 B21  0   0
     * |  B12 B22  0   0
     * |  ...
     * +-- at index offset \sum_{i<s} nMonomersPaddedInGroup[i] * MAX_CONNECTIVITY
     *     which is NOT equal to iSubGroupOffset[s] * MAX_CONNECTIVITY if the
     *     padding is different per species!
     * @endverbatim
     *
     * where Aij denotes the j-th neighbor of monomer i, such that parallel
     * access over all monomers of one species to the 1st,2nd,... neighbor
     * is linear in memory.
     * Having A11, A21, B11 aligned instead of only A11,B11,C11,... would be
     * optimal.
     *
     * Because the alignment of the PolymerSystem has more data, its alignment
     * is a harder condition than the alignment of mNeighbors, therefore we
     * don't have to recalculate everything again, and instead use the
     * alignments given as number of elements in iSubGroupOffset.
     * But the offsets are not enough, better would be a species-wise padding
     * number, i.e. nMonomersPaddedInGroup[i] = iSubGroupOffset[s+1] -
     * iSubGroupOffset[s] and the last entry would have to be recalculated
     *   -> just set this in the same loop where iSubGroupOffset is calculated
     *
     * Therefore the access to the j-th neighbor of monomer i of species s
     * would be ... too complicated, I need a new class for this problem.
     */
    MirroredVector < T_Id    > * mNeighborsSorted;
    MirroredVector < uint8_t > * mNeighborsSortedSizes;
    AlignedMatrices< T_Id    >   mNeighborsSortedInfo;
//     /**
//      * Difficult to merge this with mNeighbors as MirroredVector does not
//      * support too complicated data structures, i.e. we can't use MonomerEdges
//      * which in turn means we would have to do everything manually, especially
//      * changing the call to the graphColoring would be difficult and so on ...
//      * This is needed to recalculate mNeighborsSorted on GPU after resorting
//      * the monomers!
//      */
//     MirroredVector < T_Id   > * mNeighborsUnsorted;
//     seems not to be used anymore -_o

    RandomNumberGenerators randomNumbers;

    bool    mUsePeriodicMonomerSorting;
    int64_t mnStepsBetweenSortings;

    
    int64_t mAge;
    bool mIsPeriodicX;
    bool mIsPeriodicY;
    bool mIsPeriodicZ;


    BondVectorSet checkBondVector;
    T_BoxSize mBoxX     ;
    T_BoxSize mBoxY     ;
    T_BoxSize mBoxZ     ;
    BoxCheck boxCheck   ; 
    //holds some methods which can be set before usage of the GPU..
    Method met;
    T_BoxSize mBoxXM1   ;
    T_BoxSize mBoxYM1   ;
    T_BoxSize mBoxZM1   ;
    T_BoxSize mBoxXLog2 ;
    T_BoxSize mBoxXYLog2;
    uint32_t hGlobalIterator; // used for the RNG, equal to mAge + iStep * nSpecies + iSubstep
    int            miGpuToUse;
    cudaDeviceProp mCudaProps;
    uint8_t mnSplitColors;
    bool diagMovesOn;
 
public:
    UpdaterGPUScBFM();
    virtual ~UpdaterGPUScBFM();
    void destruct();

    /**
     * all these setter methods are quite finicky in how they are to be used!
     * Dependencies:
     *   setGpu                 : None
     *   copyBondSet            : None
     *   setLatticeSize         : None
     *   setNrOfAllMonomers     : setGpu
     *   setAttribute           : setNrOfAllMonomers
     *   setMonomerCoordinates  : setNrOfAllMonomers
     *   setConnectivity        : setNrOfAllMonomers
     *   initialize             : copyBondSet, setAttribute, setNrOfAllMonomers, setConnectivity, setLatticeSize
     *   execute                : initialize
     * => normally setNrOfAllMonomers and setGpu schould be in the constructor ... :S
     */

protected: 
    void initializeBondTable();
    void initializeSpeciesSorting(); /* using miNewToi and miToiNew the monomers are mapped to be sorted by species */
    void initializeSpatialSorting(); /* miNewToi and miToiNew will be updated so that monomers are sorted spatially per species */
    void doSpatialSorting();
    void initializeSortedNeighbors();
    void initializeSortedMonomerPositions();
    void initializeLattices();
    void checkMonomerReorderMapping();
    void findAndRemoveOverflows( bool copyToHost = true );
    /**
     * Checks for excluded volume condition and for correctness of all monomer bonds
     */
    void checkSystem() const;
    void checkLatticeOccupation() const ;
    void checkBonds() const ;
    void doCopyBack();
    void doCopyBackConnectivity(); 
    void doCopyBackMonomerPositions();
    
    template< int MoveSize > void launch_CheckSpecies          (const size_t nBlocks, const size_t nThreads, const size_t iSpecies, const size_t iOffsetLatticeTmp, const uint64_t seed );
    void launch_CheckSpeciesWithMonomericMoveType          (const size_t nBlocks, const size_t nThreads, const size_t iSpecies, const size_t iOffsetLatticeTmp, const uint64_t seed, cudaTextureObject_t const texAllowedToMoveInSpecies);
    template< int MoveSize > void launch_CheckReactiveSpecies  (const size_t nBlocks, const size_t nThreads, const size_t iSpecies, const size_t iOffsetLatticeTmp, const uint64_t seed, uint32_t AASpeciesFlag, cudaTextureObject_t const texAllowedToMoveInSpecies);
    void launch_PerformSpecies        (const size_t nBlocks, const size_t nThreads, const size_t iSpecies, cudaTextureObject_t texLatticeTmp );
    void launch_PerformSpeciesAndApply(const size_t nBlocks, const size_t nThreads, const size_t iSpecies, cudaTextureObject_t texLatticeTmp );
    void launch_ZeroArraySpecies      (const size_t nBlocks, const size_t nThreads, const size_t iSpecies );
    void launch_CountFilteredCheck    (const size_t nBlocks, const size_t nThreads, const size_t iSpecies, cudaTextureObject_t texLatticeTmp, unsigned long long int * dpFiltered , const size_t iOffsetLatticeTmp);
    void launch_countFilteredPerform  (const size_t nBlocks, const size_t nThreads, const size_t iSpecies, cudaTextureObject_t texLatticeTmp, unsigned long long int * dpFiltered );

public:

    void initialize();
    inline bool execute(){ return true; }
    void cleanup();

    /* setter methods */
    void setGpu               ( int iGpuToUse );
    void copyBondSet          ( int dx, int dy, int dz, bool bondForbidden );
    void setShearForce        ( double shearForce );
    void setNrOfAllMonomers   ( T_Id nAllMonomers );
    void setAutoColoring      ( bool bSetAutoColoring_);
    void setAttributeTag      ( T_Id i, int32_t attribute ); // this is to be NOT the coloring as needed for parallelizing the BFM, it is to be used for additional e.g. physical attributes like actual chemical types
    void setMonomerCoordinates( T_Id i, T_Coordinate x, T_Coordinate y, T_Coordinate z );
    void setConnectivity      ( T_Id monoidx1, T_Id monoidx2 );
    void setLatticeSize       ( T_BoxSize boxX, T_BoxSize boxY, T_BoxSize boxZ );
    void setDiagonalMovesOn   ( bool diagMovesOn_ ); 
    void setMethod(Method& met_){met=met_;}
    /* how often to double the initial number of colors */
    inline void setSplitColors( uint8_t const rnSplitColors ){ mnSplitColors = rnSplitColors; };
    void runSimulationOnGPU  ( uint32_t nrMCS_per_Call );
    
    uint32_t getNrOfAllMonomers(); 
    int32_t  getAttributeTag(T_Id i);
    uint32_t getNumLinks(uint32_t MonID);
    uint32_t getNeighborIdx(uint32_t MonID, uint32_t BondID);
    /* using T_Coordinate with int64_t throws error as LeMonADE itself is limited to 32 bit positions! */
    int32_t getMonomerPositionInX( T_Id i );
    int32_t getMonomerPositionInY( T_Id i );
    int32_t getMonomerPositionInZ( T_Id i );
    Method getMethod(){ return met;}
    void setPeriodicity( bool isPeriodicX, bool isPeriodicY, bool isPeriodicZ );
    inline void    setAge( int64_t rAge ){ mAge = rAge; }
    inline int64_t getAge( void ) const{ return mAge; }

    /* this is a performance feature, but one which also changes the order
     * in which random numbers are generated making binary comparisons of
     * the results moot */
    inline void    setStepsBetweenSortings( int64_t rnStepsBetweenSortings )
    {
        mUsePeriodicMonomerSorting = rnStepsBetweenSortings > 0;
        mnStepsBetweenSortings = rnStepsBetweenSortings;
    }
    inline int64_t getStepsBetweenSortings( void ) const{ return mnStepsBetweenSortings; }
};
