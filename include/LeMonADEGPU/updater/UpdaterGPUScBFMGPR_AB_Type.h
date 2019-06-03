/*
 * UpdaterGPUScBFMGPR_AB_Type.h
 *
 *  Created on: 27.07.2017
 *      Author: Ron Dockhorn
 */

#pragma once


#include <cassert>
#include <chrono>                           // high_resolution_clock
#include <cstdio>                           // printf
#include <cstdint>                          // uint32_t, size_t
#include <stdexcept>

#include <cuda_runtime_api.h>               // cudaStream_t, cudaDeviceProp
#include <LeMonADE/utility/RandomNumberGenerators.h>


#include <LeMonADEGPU/utility/cudacommon.hpp>
#include <LeMonADEGPU/utility/SelectiveLogger.hpp>
#include <LeMonADEGPU/utility/alignedMatrizes.h>

/* This is still used, at least the 32 bit version! */
#if 0
    typedef uint32_t uintCUDA;
    typedef int32_t  intCUDA;
    #define MASK5BITS 0x7FFFFFE0
#else
    typedef uint16_t uintCUDA;
    typedef int16_t  intCUDA;
    #define MASK5BITS 0x7FE0
#endif
using vecIntCUDA = CudaVec4< intCUDA >::value_type;


#define MAX_CONNECTIVITY 4 // original connectivity
// #define MAX_CONNECTIVITY 8 // needed for the coloring example

#define NONPERIODICITY


class UpdaterGPUScBFMGPR_AB_Type
{
private:
    SelectedLogger mLog;

    cudaStream_t mStream;

    RandomNumberGenerators randomNumbers;

    bool mForbiddenBonds[512];
    //int BondAsciiArray[512];

    uint32_t nAllMonomers;
    
    uint32_t nRingMonomers;
    uint32_t nChains;
    uint32_t nMonomersPerChain;
    /**
     * Vector of length boxX * boxY * boxZ. Actually only contains 0 if
     * the cell / lattice point is empty or 1 if it is occupied by a monomer
     * Suggestion: bitpack it to save 8 times memory and possibly make the
     *             the reading faster if it is memory bound ???
     */
    uint8_t * mLattice; // being used for checkLattice nothing else ...
    MirroredTexture< uint8_t > * mLatticeOut, * mLatticeTmp;

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
    std::vector< intCUDA > mPolymerSystem;
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
    MirroredVector< vecIntCUDA > * mPolymerSystemSorted;
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
    

    
public:
    using T_Flags = uint8_t; // uint16_t, uint32_t
private:
    MirroredVector< T_Flags > * mPolymerFlags;

    static auto constexpr nBytesAlignment    = 512u;
    static auto constexpr nElementsAlignment = nBytesAlignment / ( 4u * sizeof( intCUDA ) );
    static_assert( nBytesAlignment == nElementsAlignment * 4u * sizeof( intCUDA), "Element type of polymer systems seems to be larger than the Bytes we need to align on!" );

    /* for each monomer the attribute 1 (A) or 2 (B) is stored
     * -> could be 1 bit per monomer ... !!!
     * @see http://www.gotw.ca/gotw/050.htm
     * -> wow std::vector<bool> already optimized for space with bit masking!
     * This is only needed once for initializing mMonomerIdsA,B */
    int32_t * mAttributeSystem;
    std::vector< uint8_t > mGroupIds; /* for each monomer stores the color / attribute / group ID/tag */
    std::vector< uint8_t > mRingGroupIds; /* for each monomer stores the color / attribute / group ID/tag */
    std::vector< size_t > mnElementsInGroup;
    std::vector< size_t > iToiNew;   /* for each old monomer stores the new position */
    std::vector< size_t > iNewToi;   /* for each new monomer stores the old position */
    std::vector< size_t > iSubGroupOffset; /* stores offsets (in number of elements not bytes) to each aligned subgroup vector in mPolymerSystemSorted */

    /* needed to decide whether we can even check autocoloring with given one */
    bool bSetAttributeCalled;
    /* in order to decide how to work with setMonomerCoordinates and
     * getMonomerCoordinates. This way we might just return the monomer
     * info directly instead of needing to desort after each execute run */
    bool bPolymersSorted;

public:
    /* stores amount and IDs of neighbors for each monomer */
    struct MonomerEdges
    {
        uint32_t size;
        uint32_t neighborIds[ MAX_CONNECTIVITY ];
    };
    /* size is encoded in mPolymerSystem to make things faster */
    struct MonomerEdgesCompressed
    {
        uint32_t neighborIds[ MAX_CONNECTIVITY ];
    };
private:
    std::vector< MonomerEdges > mNeighbors;
    /**
     * stores the IDs of all neighbors as is needed to check for the bond
     * set / length restrictions.
     * But after the sorting of mPolymerSystem the IDs also changed.
     * And as I don't want to push iToiNew and iNewToi to the GPU instead
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
    //! this is only used for the coloring of the graph 
    std::vector< MonomerEdges > mNeighborsRings;
    MirroredVector < uint32_t > * mNeighborsSorted;
    MirroredVector < uint8_t  > * mNeighborsSortedSizes;
    AlignedMatrices< uint32_t >   mNeighborsSortedInfo;
    /** 
     * The system contains linear chains with sinlge monomers randomly 
     * attatched to it. For these single monomers it is allowd to change the 
     * bond to the chain they are attatched to. Meaning they slide along the 
     * chain backbone. For that purpose we need to know the maximum number of 
     * bonds which are allowed for the chain ends (2),  monomers of the remaining
     * chain (3) and the ring monomers (2).
     */
    MirroredVector< uint8_t > * mMonomerStructureTag; 

    uint32_t   mBoxX     ;
    uint32_t   mBoxY     ;
    uint32_t   mBoxZ     ;
    uint32_t   mBoxXM1   ;
    uint32_t   mBoxYM1   ;
    uint32_t   mBoxZM1   ;
    uint32_t   mBoxXLog2 ;
    uint32_t   mBoxXYLog2;
    uint32_t   mGlobalIterator;
    bool excludedVolumeON;
    bool RingSlidingON;
    bool DiagonalMovesON;
    bool mBoxXIsPeriodic;
    bool mBoxYIsPeriodic;
    bool mBoxZIsPeriodic;

    int            miGpuToUse;
    cudaDeviceProp mCudaProps;


    /**
     * If we constrict each index to 1024=2^10 which already is quite large,
     * 256=2^8 being normally large, then this means that the linearzed index
     * should have a range of 2^30, meaning uint32_t as output is pretty
     * fixed with uint16_t being way too few bits
     */
    uint32_t linearizeBoxVectorIndex
    (
        uint32_t const & ix,
        uint32_t const & iy,
        uint32_t const & iz
    );

    /**
     * Checks for excluded volume condition and for correctness of all monomer bonds
     */
    void checkSystem();

public:
    UpdaterGPUScBFMGPR_AB_Type();
    virtual ~UpdaterGPUScBFMGPR_AB_Type();
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

    void initialize();
    inline bool execute(){ return true; }
    void cleanup();

    /* setter methods */
    void setGpu                  ( int iGpuToUse );
    void setEVON                 ( bool excludedVolumeON );
    void setRingSliding          ( bool RingSlidingON );
    void setDiagonalMovesON      ( bool DiagonalMovesON);
    void setForce                ( double shearForce);
    void copyBondSet             ( int dx, int dy, int dz, bool bondForbidden );
    void setNrOfAllMonomers      ( uint32_t nAllMonomers );
    void setNrOfRingMonomers     ( uint32_t nRingMonomers );
    void setNrOfChains           ( uint32_t nChains );
    void setNrOfMonomersPerChain ( uint32_t nMonomersPerChain );
    void setAttribute            ( uint32_t i, int32_t attribute );
    void setMonomerCoordinates   ( uint32_t i, int32_t x, int32_t y, int32_t z );
    void setConnectivity         ( uint32_t monoidx1, uint32_t monoidx2 );
    void setRingConnectivity     ( uint32_t monoidx1, uint32_t monoidx2 );
    void setLatticeSize          ( uint32_t boxX, uint32_t boxY, uint32_t boxZ );

    /**
     * sets monomer positions given in mPolymerSystem in mLattice to occupied
     */
    void populateLattice();
    void runSimulationOnGPU( int32_t nrMCS_per_Call );

    int32_t  getMonomerPositionInX ( uint32_t i                    );
    int32_t  getMonomerPositionInY ( uint32_t i                    );
    int32_t  getMonomerPositionInZ ( uint32_t i                    );
    uint32_t getConnectivity       ( uint32_t i, uint32_t iNeighbor);
    uint32_t getMaximumConnectivity( uint32_t i                    );

    void setPeriodicity( bool isPeriodicX, bool isPeriodicY, bool isPeriodicZ );

    /* for benchmarking purposes */
    std::chrono::time_point< std::chrono::high_resolution_clock > mtCopyBack0;

// private:
// //Helper functions for different RNGs
//     void runSimulationOnGPU_hash( int32_t nrMCS_per_Call );
//     void runSimulationOnGPU_CuRandPhilox( int32_t nrMCS_per_Call );
//     void runSimulationOnGPU_pcg( int32_t nrMCS_per_Call );
//     void runSimulationOnGPU_curand( int32_t nrMCS_per_Call );
//     void runSimulationOnGPU_saru( int32_t nrMCS_per_Call );
};
