/*
 * UpdaterGPUScBFM.cpp
 *
 *  Created on: 27.07.2017
 *      Authors: Ron Dockhorn, Maximilian Knespel
 */


#include <LeMonADEGPU/updater/UpdaterGPUScBFM.h>


#include <algorithm>                        // fill, sort, count
#include <chrono>                           // std::chrono::high_resolution_clock
#include <cstdio>                           // printf
#include <cstdlib>                          // exit
#include <cstring>                          // memset
#include <ctime>
#include <functional>
#include <iostream>
#include <limits>                           // numeric_limits
#include <stdexcept>
#include <stdint.h>
#include <sstream>
#include <vector>
#include <type_traits>                      // make_unsigned

#include <cuda_profiler_api.h>              // cudaProfilerStop
#include <thrust/execution_policy.h>        // thrust::seq, thrust::host
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/gather.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/sort.h>                    // sort_by_key

//extract only the things which are really needed from the below two files:
#include <extern/Fundamental/BitsCompileTime.hpp>
#include <LeMonADEGPU/utility/cudacommon.hpp>

#include <LeMonADEGPU/utility/SelectiveLogger.hpp>
#include <LeMonADEGPU/utility/graphColoring.tpp>
#include <LeMonADEGPU/core/rngs/Saru.h>
#include <LeMonADEGPU/core/MonomerEdges.h>
#include <LeMonADEGPU/core/constants.cuh>
#include <LeMonADEGPU/feature/BoxCheck.h>
#include <LeMonADEGPU/core/Method.h>

#include <LeMonADEGPU/utility/DeleteMirroredObject.h>
#include <LeMonADEGPU/core/BondVectorSet.h>

/* shorten full type names for kernels (assuming these are independent of the template parameter) */
using T_Flags            = UpdaterGPUScBFM< uint8_t >::T_Flags      ;
using T_Lattice          = UpdaterGPUScBFM< uint8_t >::T_Lattice    ;
using T_Coordinate       = UpdaterGPUScBFM< uint8_t >::T_Coordinate ;
using T_Coordinates      = UpdaterGPUScBFM< uint8_t >::T_Coordinates;
using T_Id               = UpdaterGPUScBFM< uint8_t >::T_Id         ;
using getBitPackedTextureFunction = UpdaterGPUScBFM<uint8_t>::getBitPackedTextureFunction;
__device__ getBitPackedTextureFunction functor = &BitPacking::bitPackedTextureGetStandard;
// typedef  T_Lattice (BitPacking::*getBitPackedTextureFunction)(cudaTextureObject_t tex, int i) const ; 

/* Since CUDA 5.5 (~2014) there do exist texture objects which are much
 * easier and can actually be used as kernel arguments!
 * @see https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-kepler-texture-objects-improve-performance-and-flexibility/
 * "What is not commonly known is that each outstanding texture reference that
 *  is bound when a kernel is launched incurs added launch latency—up to 0.5 μs
 *  per texture reference. This launch overhead persists even if the outstanding
 *  bound textures are not even referenced by the kernel. Again, using texture
 *  objects instead of texture references completely removes this overhead."
 * => they only exist for kepler -.- ...
 */

/**
 * Checks the 3x3 grid one in front of the new position in the direction of the
 * move given by axis.
 *
 * @verbatim
 *           ____________
 *         .'  .'  .'  .'|
 *        +---+---+---+  +     y
 *        | 6 | 7 | 8 |.'|     ^ z
 *        +---+---+---+  +     |/
 *        | 3/| 4/| 5 |.'|     +--> x
 *        +-/-+-/-+---+  +
 *   0 -> |+---+1/| 2 |.'  ^          ^
 *        /|/-/|/-+---+   /          / axis direction +z (axis = 0b101)
 *       / +-/-+         /  2 (*dz) /                              ++|
 *      +---+ /         /                                         /  +/-
 *      |/X |/         L                                        xyz
 *      +---+  <- X ... current position of the monomer
 * @endverbatim
 *
 * @param[in] axis +-x, +-y, +-z in that order from 0 to 5, or put in another
 *                 equivalent way: the lowest bit specifies +(1) or -(0) and the
 *                 Bit 2 and 1 specify the axis: 0b00=x, 0b01=y, 0b10=z
 * @return Returns true if any of that is occupied, i.e. if there
 *         would be a problem with the excluded volume condition.
 * @todo replace CALL_MEMBER_FN macro with template 
 */

#define CALL_MEMBER_FN(object,ptrToMember)  ((object).*(ptrToMember))
__device__ inline bool checkFront
(
    cudaTextureObject_t const & texLattice,
    uint32_t            const & x0        ,
    uint32_t            const & y0        ,
    uint32_t            const & z0        ,
    T_Flags             const & axis      ,
    Method		const & met       ,
    getBitPackedTextureFunction func = &BitPacking::bitPackedTextureGetStandard, 
    T_Id              * const   iOldPos = NULL
)
{
   
    auto const x0MDX  = met.getCurve().linearizeBoxVectorIndexX( x0 - uint32_t(1) );
    auto const x0Abs  = met.getCurve().linearizeBoxVectorIndexX( x0               );
    auto const x0PDX  = met.getCurve().linearizeBoxVectorIndexX( x0 + uint32_t(1) );
    auto const y0MDY  = met.getCurve().linearizeBoxVectorIndexY( y0 - uint32_t(1) );
    auto const y0Abs  = met.getCurve().linearizeBoxVectorIndexY( y0               );
    auto const y0PDY  = met.getCurve().linearizeBoxVectorIndexY( y0 + uint32_t(1) );
    auto const z0MDZ  = met.getCurve().linearizeBoxVectorIndexZ( z0 - uint32_t(1) );
    auto const z0Abs  = met.getCurve().linearizeBoxVectorIndexZ( z0               );
    auto const z0PDZ  = met.getCurve().linearizeBoxVectorIndexZ( z0 + uint32_t(1) );

    auto const dx = DXTable_d[ axis ];   // 2*axis-1
    auto const dy = DYTable_d[ axis ];   // 2*(axis&1)-1
    auto const dz = DZTable_d[ axis ];   // 2*(axis&1)-1

    if ( iOldPos != NULL )
        *iOldPos = x0Abs + y0Abs + z0Abs;

    uint32_t is[9];
    switch ( axis >> 1 )
    {
	case 0: is[7] = met.getCurve().linearizeBoxVectorIndexX( x0 + dx + dx ); 
		/* this line adds all three z directions */
		is[2]  = is[7] + z0MDZ; is[5]  = is[7] + z0Abs; is[8]  = is[7] + z0PDZ;
		/* now for all three results we generate all 3 different y positions */
		is[0]  = is[2] + y0MDY; is[1]  = is[2] + y0Abs; is[2] +=         y0PDY;
		is[3]  = is[5] + y0MDY; is[4]  = is[5] + y0Abs; is[5] +=         y0PDY;
		is[6]  = is[8] + y0MDY; is[7]  = is[8] + y0Abs; is[8] +=         y0PDY;
		break;
		/**
		* so the order for the 9 positions when moving in x direction is:
		* @verbatim
		* z ^
		*   | 0 1 2
		*   | 3 4 5
		*   | 6 7 8
		*   +------> y
		* @endverbatim
		*/
	case 1: is[7] = met.getCurve().linearizeBoxVectorIndexY( y0 + dy + dy ); 
		is[2]  = is[7] + z0MDZ; is[5]  = is[7] + z0Abs; is[8]  = is[7] + z0PDZ;
		is[0]  = is[2] + x0MDX; is[1]  = is[2] + x0Abs; is[2] +=         x0PDX;
		is[3]  = is[5] + x0MDX; is[4]  = is[5] + x0Abs; is[5] +=         x0PDX;
		is[6]  = is[8] + x0MDX; is[7]  = is[8] + x0Abs; is[8] +=         x0PDX;
		break;
		/**
		* @verbatim
		* z ^
		*   | 0 1 2
		*   | 3 4 5
		*   | 6 7 8
		*   +------> x
		* @endverbatim
		*/
	case 2: is[7] = met.getCurve().linearizeBoxVectorIndexZ( z0 + dz + dz ); 
		is[2]  = is[7] + y0MDY; is[5]  = is[7] + y0Abs; is[8]  = is[7] + y0PDY;
		is[0]  = is[2] + x0MDX; is[1]  = is[2] + x0Abs; is[2] +=         x0PDX;
		is[3]  = is[5] + x0MDX; is[4]  = is[5] + x0Abs; is[5] +=         x0PDX;
		is[6]  = is[8] + x0MDX; is[7]  = is[8] + x0Abs; is[8] +=         x0PDX;
		break;
		/**
		* @verbatim
		* y ^
		*   | 0 1 2
		*   | 3 4 5
		*   | 6 7 8
		*   +------> x
		* @endverbatim
		*/
    }
    return  CALL_MEMBER_FN(met.getPacking(), func)( texLattice, is[ 0 ] ) +
	    CALL_MEMBER_FN(met.getPacking(), func)( texLattice, is[ 1 ] ) +
	    CALL_MEMBER_FN(met.getPacking(), func)( texLattice, is[ 2 ] ) +
	    CALL_MEMBER_FN(met.getPacking(), func)( texLattice, is[ 3 ] ) +
	    CALL_MEMBER_FN(met.getPacking(), func)( texLattice, is[ 4 ] ) +
	    CALL_MEMBER_FN(met.getPacking(), func)( texLattice, is[ 5 ] ) +
	    CALL_MEMBER_FN(met.getPacking(), func)( texLattice, is[ 6 ] ) +
	    CALL_MEMBER_FN(met.getPacking(), func)( texLattice, is[ 7 ] ) +
	    CALL_MEMBER_FN(met.getPacking(), func)( texLattice, is[ 8 ] ) ;

}

/**
 * @brief checks all bonds of the current monomer
 * @return true if bond is forbidden, false if all bonds are allowed
 */
template< typename T_UCoordinateCuda >
__device__   bool checkNeighboringBonds(
uint8_t     const * const              dpNeighborsSizes        ,
T_Id                const              iMonomer                , 
T_Id        const * const              dpNeighbors             ,
uint32_t            const              rNeighborsPitchElements ,
typename CudaVec4< T_UCoordinateCuda >::value_type
	    const * const __restrict__ dpPolymerSystem         ,
typename CudaVec4< T_UCoordinateCuda >::value_type const     r1,
BondVectorSet       const              checkBondVector
){
  /* check whether the new position would result in invalid bonds
  * between this monomer and its neighbors */
  auto const nNeighbors = dpNeighborsSizes[ iMonomer ];
  for ( auto iNeighbor = decltype( nNeighbors )(0); iNeighbor < nNeighbors; ++iNeighbor )
  {
      auto const iGlobalNeighbor = dpNeighbors[ iNeighbor * rNeighborsPitchElements + iMonomer ];
      auto const data2 = dpPolymerSystem[ iGlobalNeighbor ];
      if ( checkBondVector( data2.x - r1.x, data2.y - r1.y, data2.z - r1.z ) )
      {
	return true; 
      }
  }
  return false;	
}
#define MAKEINSTANCE(TYPE) \
template __device__  bool checkNeighboringBonds < TYPE > (uint8_t const * const, T_Id const, T_Id const * const, uint32_t const, typename CudaVec4< TYPE >::value_type const * const __restrict__, typename CudaVec4< TYPE >::value_type const, BondVectorSet const);
MAKEINSTANCE(uint8_t);
MAKEINSTANCE(uint16_t);
MAKEINSTANCE(uint32_t);
MAKEINSTANCE(int16_t);
MAKEINSTANCE(int32_t);
#undef MAKEINSTANCE

namespace {

/**
 * Goes over all monomers of a species given specified by texSpeciesIndices
 * draws a random direction for them and checks whether that move is possible
 * with the box size and periodicity as well as the monomers at the target
 * location (excluded volume) and the new bond lengths to all neighbors.
 * If so, then the new position is set to 1 in dpLatticeTmp and encode the
 * possible movement direction in the property tag of the corresponding monomer
 * in dpPolymerSystem.
 * Note that the old position is not removed in order to correctly check for
 * excluded volume a second time.
 *
 * @param[in] rn a random number used as a kind of seed for the RNG
 * @param[in] nMonomers number of max. monomers to work on, this is for
 *            filtering out excessive threads and was prior a __constant__
 *            But it is only used one(!) time in the kernel so the caching
 *            of constant memory might not even be used.
 *            @see https://web.archive.org/web/20140612185804/http://www.pixel.io/blog/2013/5/9/kernel-arguments-vs-__constant__-variables.html
 *            -> Kernel arguments are even put into constant memory it seems:
 *            @see "Section E.2.5.2 Function Parameters" in the "CUDA 5.5 C Programming Guide"
 *            __global__ function parameters are passed to the device:
 *             - via shared memory and are limited to 256 bytes on devices of compute capability 1.x,
 *             - via constant memory and are limited to 4 KB on devices of compute capability 2.x and higher.
 *            __device__ and __global__ functions cannot have a variable number of arguments.
 * @param[in] iOffset Offste to curent species we are supposed to work on.
 *            We can't simply adjust dpPolymerSystem before calling the kernel,
 *            because we are accessing the neighbors, therefore need all the
 *            polymer data, especially for other species.
 *
 * Note: all of the three kernels do quite few work. They basically just fetch
 *       data, and check one condition and write out again. There isn't even
 *       a loop and most of the work seems to be boiler plate initialization
 *       code which could be cut if the kernels could be merged together.
 *       Why are there three kernels instead of just one
 *        -> for global synchronization
 */
template< typename T_UCoordinateCuda >
__global__ void kernelSimulationScBFMCheckSpecies
(
    typename CudaVec4< T_UCoordinateCuda >::value_type
                const * const __restrict__ dpPolymerSystem         ,
    T_Flags           * const              dpPolymerFlags          ,
    uint32_t            const              iOffset                 ,
    T_Lattice         * const __restrict__ dpLatticeTmp            ,
    T_Id        const * const              dpNeighbors             ,
    uint32_t            const              rNeighborsPitchElements ,
    uint8_t     const * const              dpNeighborsSizes        ,
    T_Id                const              nMonomers               ,
    uint64_t            const              rSeed                   ,
    uint64_t            const              rGlobalIteration        ,
    cudaTextureObject_t const              texLatticeRefOut        ,
    BoxCheck                               bCheck, 
    Method              const              met,
    BondVectorSet       const              checkBondVector
)
{
    uint32_t rn;
    int iGrid = 0;
    for ( auto iMonomer = blockIdx.x * blockDim.x + threadIdx.x;
          iMonomer < nMonomers; iMonomer += gridDim.x * blockDim.x, ++iGrid )
    {
        auto const r0 = dpPolymerSystem[ iOffset + iMonomer ];
        /* upcast int3 to int4 in preparation to use PTX SIMD instructions */
        //int4 const r0 = { r0Raw.x, r0Raw.y, r0Raw.z, 0 }; // not faster nor slower
        //select random direction. Own implementation of an rng :S? But I think it at least# was initialized using the LeMonADE RNG ...
        if ( iGrid % 1 == 0 ) // 12 = floor( log(2^32) / log(6) )
        {
	  Saru rng(rGlobalIteration,iMonomer,rSeed);
	  rn =rng.rng32();
        }

        int direction = rn % 6;

         /* select random direction. Do this with bitmasking instead of lookup ??? */
        typename CudaVec4< T_UCoordinateCuda >::value_type const r1 = {
            T_UCoordinateCuda( r0.x + DXTable_d[ direction ] ),
            T_UCoordinateCuda( r0.y + DYTable_d[ direction ] ),
            T_UCoordinateCuda( r0.z + DZTable_d[ direction ] )
        };

	if (    bCheck(r1.x,r1.y,r1.z) && 
	      ! checkNeighboringBonds<T_UCoordinateCuda>(dpNeighborsSizes, iMonomer, dpNeighbors, rNeighborsPitchElements, dpPolymerSystem, r1, checkBondVector ) && 
	      ! checkFront( texLatticeRefOut, r0.x, r0.y, r0.z, direction, met, &BitPacking::bitPackedTextureGetStandard ) )
	{
	    /* everything fits so perform move on temporary lattice */
	    /* can I do this ??? dpPolymerSystem is the device pointer to the read-only
	      * texture used above. Won't this result in read-after-write race-conditions?
	      * Then again the written / changed bits are never used in the above code ... */
	    direction += T_Flags(8) /* can-move-flag */;
	    met.getPacking().bitPackedSet(dpLatticeTmp, met.getCurve().linearizeBoxVectorIndex( r1.x, r1.y, r1.z ));
	}

        dpPolymerFlags[ iMonomer ] = direction;
    }
}
template< typename T_UCoordinateCuda >
__global__ void kernelSimulationScBFMCheckReactiveSpecies
(
    typename CudaVec4< T_UCoordinateCuda >::value_type
                const * const __restrict__ dpPolymerSystem         ,
    T_Flags           * const              dpPolymerFlags          ,
    uint32_t            const              iOffset                 ,
    T_Lattice         * const __restrict__ dpLatticeTmp            ,
    T_Id        const * const              dpNeighbors             ,
    uint32_t            const              rNeighborsPitchElements ,
    uint8_t     const * const              dpNeighborsSizes        ,
    T_Id                const              nMonomers               ,
    uint64_t            const              rSeed                   ,
    uint64_t            const              rGlobalIteration        ,
    cudaTextureObject_t const              texLatticeRefOut        ,
    BoxCheck                               boxCheck                , 
    Method              const              met                     ,
    BondVectorSet       const              checkBondVector         ,
    cudaTextureObject_t const              texAllowedToMove	   ,
    uint32_t            const              AASpeciesFlag
)
{
    uint32_t rn;
    int iGrid = 0;
    for ( auto iMonomer = blockIdx.x * blockDim.x + threadIdx.x;
          iMonomer < nMonomers; iMonomer += gridDim.x * blockDim.x, ++iGrid )
    {

	  
	/* upcast int3 to int4 in preparation to use PTX SIMD instructions */
	//int4 const r0 = { r0Raw.x, r0Raw.y, r0Raw.z, 0 }; // not faster nor slower
	//select random direction. Own implementation of an rng :S? But I think it at least# was initialized using the LeMonADE RNG ...
	if ( iGrid % 1 == 0 ) // 12 = floor( log(2^32) / log(6) )
	{
	  Saru rng(rGlobalIteration,iMonomer,rSeed);
	  rn =rng.rng32();
	}
	
	int direction = rn % 6;
	
	if (tex1Dfetch<uint8_t>( texAllowedToMove, iMonomer) == AASpeciesFlag )
	{  
	  auto const r0 = dpPolymerSystem[ iOffset + iMonomer ];
	  /* select random direction. Do this with bitmasking instead of lookup ??? */
	  typename CudaVec4< T_UCoordinateCuda >::value_type const r1 = {
	      T_UCoordinateCuda( r0.x + DXTable_d[ direction ] ),
	      T_UCoordinateCuda( r0.y + DYTable_d[ direction ] ),
	      T_UCoordinateCuda( r0.z + DZTable_d[ direction ] )
	  };
	  if (    boxCheck(r1.x,r1.y,r1.z) &&
		! checkNeighboringBonds<T_UCoordinateCuda>(dpNeighborsSizes, iMonomer, dpNeighbors, rNeighborsPitchElements, dpPolymerSystem, r1, checkBondVector ) &&
		! checkFront( texLatticeRefOut, r0.x, r0.y, r0.z, direction, met, &BitPacking::bitPackedTextureGetStandard ) 
	     )
	  {
	      /* everything fits so perform move on temporary lattice */
	      /* can I do this ??? dpPolymerSystem is the device pointer to the read-only
		* texture used above. Won't this result in read-after-write race-conditions?
		* Then again the written / changed bits are never used in the above code ... */
	      direction += T_Flags(8) /* can-move-flag */;
	      met.getPacking().bitPackedSet(dpLatticeTmp, met.getCurve().linearizeBoxVectorIndex( r1.x, r1.y, r1.z ));
	  }
	}
	dpPolymerFlags[ iMonomer ] = direction;        
    }
}

/*
colordiff <( sed -n 931,1028p ../src/pscbfm/UpdaterGPUScBFM.cu ) <( sed -n 1031,1090p ../src/pscbfm/UpdaterGPUScBFM.cu )
*/
template< typename T_UCoordinateCuda >
__global__ void kernelCountFilteredCheck
(
    typename CudaVec4< T_UCoordinateCuda >::value_type
                     const * const __restrict__ dpPolymerSystem        ,
    T_Flags          const * const              dpPolymerFlags         ,
    uint32_t                 const              iOffset                ,
    T_Lattice        const * const __restrict__ /* dpLatticeTmp */     ,
    T_Id             const * const              dpNeighbors            ,
    uint32_t                 const              rNeighborsPitchElements,
    uint8_t          const * const              dpNeighborsSizes       ,
    T_Id                     const              nMonomers              ,
    cudaTextureObject_t      const              texLatticeRefOut       ,
    unsigned long long int * const              dpFiltered             ,
    BoxCheck                                    bCheck		       ,
    Method                                      met,
    BondVectorSet            const              checkBondVector
)
{
    for ( auto iMonomer = blockIdx.x * blockDim.x + threadIdx.x;
          iMonomer < nMonomers; iMonomer += gridDim.x * blockDim.x )
    {
        auto const r0 = dpPolymerSystem[ iOffset + iMonomer ];
        auto const properties = dpPolymerFlags[ iMonomer ];
        auto const direction = properties & T_Flags(7); // 7=0b111

        typename CudaVec4< T_UCoordinateCuda >::value_type const r1 = {
            T_UCoordinateCuda( r0.x + DXTable_d[ direction ] ),
            T_UCoordinateCuda( r0.y + DYTable_d[ direction ] ),
            T_UCoordinateCuda( r0.z + DZTable_d[ direction ] )
        };
	if ( bCheck(r1.x,r1.y,r1.z) &&
	     checkFront( texLatticeRefOut, r0.x, r0.y, r0.z, direction, met, &BitPacking::bitPackedTextureGetStandard  ) )
	{
	    atomicAdd( dpFiltered+2, 1ull );
	    if ( ! checkNeighboringBonds<T_UCoordinateCuda>(dpNeighborsSizes, iMonomer, dpNeighbors, rNeighborsPitchElements, dpPolymerSystem, r1, checkBondVector ) ) /* this is the more real relative use-case where invalid bonds are already filtered out */
		atomicAdd( dpFiltered+3, 1ull );
	}

    }
}


/**
 * Recheck whether the move is possible without collision, using the
 * temporarily parallel executed moves saved in texLatticeTmp. If so,
 * do the move in dpLattice. (Still not applied in dpPolymerSystem!)
 */
template< typename T_UCoordinateCuda >
__global__ void kernelSimulationScBFMPerformSpecies
(
    typename CudaVec4< T_UCoordinateCuda >::value_type
                        const * const              dpPolymerSystem,
    T_Flags                   * const              dpPolymerFlags ,
    T_Lattice                 * const __restrict__ dpLattice      ,
    T_Id                        const              nMonomers      ,
    cudaTextureObject_t         const              texLatticeTmp  ,
    Method                      const              met 
)
{
    for ( auto iMonomer = blockIdx.x * blockDim.x + threadIdx.x;
          iMonomer < nMonomers; iMonomer += gridDim.x * blockDim.x )
    {
        auto const properties = dpPolymerFlags[ iMonomer ];
        if ( ( properties & T_Flags(8) ) == T_Flags(0) ) // impossible move
            continue;

        auto const r0 = dpPolymerSystem[ iMonomer ];
        //uint3 const r0 = { r0Raw.x, r0Raw.y, r0Raw.z }; // slower
        auto const direction = properties & T_Flags(7); // 7=0b111
        uint32_t iOldPos;
	//if check Front is true (there is a monomer) : go to next monomer in the grid 
        if ( checkFront( texLatticeTmp, r0.x, r0.y, r0.z, direction, met,  &BitPacking::bitPackedTextureGet, &iOldPos ) )
	  continue;

        /* If possible, perform move now on normal lattice */
        dpPolymerFlags[ iMonomer ] = properties | T_Flags(16); // indicating allowed move
        dpLattice[ iOldPos ] = 0;
        dpLattice[ met.getCurve().linearizeBoxVectorIndex( r0.x + DXTable_d[ direction ],
                                            r0.y + DYTable_d[ direction ],
                                            r0.z + DZTable_d[ direction ] ) ] = 1;
        /* We can't clean the temporary lattice in here, because it still is being
         * used for checks. For cleaning we need only the new positions.
         * But we can't use the applied positions, because we also need to clean
         * those particles which couldn't move in this second kernel but where
         * still set in the lattice by the first kernel! */
    }
}

template< typename T_UCoordinateCuda >
__global__ void kernelSimulationScBFMPerformSpeciesAndApply
(
    typename CudaVec4< T_UCoordinateCuda >::value_type
                        * const              dpPolymerSystem,
    T_Flags             * const              dpPolymerFlags ,
    T_Lattice           * const __restrict__ dpLattice      ,
    T_Id                  const              nMonomers      ,
    cudaTextureObject_t   const              texLatticeTmp,
    Method 				     met 
)
{
    for ( auto iMonomer = blockIdx.x * blockDim.x + threadIdx.x;
          iMonomer < nMonomers; iMonomer += gridDim.x * blockDim.x )
    {
        auto const properties = dpPolymerFlags[ iMonomer ];
        if ( ! ( properties & T_Flags(8) ) ) // check if can-move flag is set
            continue;

        auto const r0 = dpPolymerSystem[ iMonomer ];
        auto const direction = properties & T_Flags(7); // 7=0b111
        uint32_t iOldPos;
        if ( checkFront( texLatticeTmp, r0.x, r0.y, r0.z, direction, met, &BitPacking::bitPackedTextureGet, &iOldPos ) )
            continue;

        /* @todo this is slower on Kepler when using DXTableUintCuda_d
         *       not sure why ... but on Pascal it might trigger
         *       uint8 calculation speedup! */
        dpLattice[ iOldPos ] = 0;
        typename CudaVec4< T_UCoordinateCuda >::value_type const r1 = {
            T_UCoordinateCuda( r0.x + DXTable_d[ direction ] ),
            T_UCoordinateCuda( r0.y + DYTable_d[ direction ] ),
            T_UCoordinateCuda( r0.z + DZTable_d[ direction ] ),
            T_UCoordinateCuda( 0 )
        };
        dpLattice[ met.getCurve().linearizeBoxVectorIndex( r1.x, r1.y, r1.z ) ] = 1;
        /* If possible, perform move now on normal lattice */
        dpPolymerSystem[ iMonomer ] = r1;
    }
}

template< typename T_UCoordinateCuda >
__global__ void kernelCountFilteredPerform
(
    typename CudaVec4< T_UCoordinateCuda >::value_type
                     const * const              dpPolymerSystem  ,
    T_Flags          const * const              dpPolymerFlags   ,
    T_Lattice        const * const __restrict__ /* dpLattice */  ,
    T_Id                     const              nMonomers        ,
    cudaTextureObject_t      const              texLatticeTmp    ,
    unsigned long long int * const              dpFiltered       ,
    Method 					met
)
{
    for ( auto iMonomer = blockIdx.x * blockDim.x + threadIdx.x;
          iMonomer < nMonomers; iMonomer += gridDim.x * blockDim.x )
    {
        auto const properties = dpPolymerFlags[ iMonomer ];
        if ( ! ( properties & T_Flags(8) ) ) // check if can-move flag is set
            continue;

        auto const r0 = dpPolymerSystem[ iMonomer ];
        auto const direction = properties & T_Flags(7); // 7=0b111
        if ( checkFront( texLatticeTmp, r0.x, r0.y, r0.z, direction, met, &BitPacking::bitPackedTextureGet, NULL ) )
            atomicAdd( dpFiltered+4, size_t(1) );
    }
}

/**
 * Apply move to dpPolymerSystem and clean the temporary lattice of moves
 * which seemed like they would work, but did clash with another parallel
 * move, unfortunately.
 * @todo it might be better to just use a cudaMemset to clean the lattice,
 *       that way there wouldn't be any memory dependencies and calculations
 *       needed, even though we would have to clean everything, instead of
 *       just those set. But that doesn't matter, because most of the threads
 *       are idling anyway ...
 *       This kind of kernel might give some speedup after stream compaction
 *       has been implemented though.
 *    -> print out how many percent of cells need to be cleaned .. I need
 *       many more statistics anyway for evaluating performance benefits a bit
 *       better!
 */
template< typename T_UCoordinateCuda >
__global__ void kernelSimulationScBFMZeroArraySpecies
(
    typename CudaVec4< T_UCoordinateCuda >::value_type
                    * const              dpPolymerSystem,
    T_Flags   const * const              dpPolymerFlags ,
    T_Lattice       * const __restrict__ dpLatticeTmp   ,
    T_Id              const              nMonomers      ,
    Method 				 met 
)
{
    for ( auto iMonomer = blockIdx.x * blockDim.x + threadIdx.x;
          iMonomer < nMonomers; iMonomer += gridDim.x * blockDim.x )
    {
        auto const properties = dpPolymerFlags[ iMonomer ];
        if ( ( properties & T_Flags(16+8) ) == T_Flags(0) ) // impossible move
            continue;

        auto r0 = dpPolymerSystem[ iMonomer ];
        auto const direction = properties & T_Flags(7); // 7=0b111

        r0.x += DXTable_d[ direction ];
        r0.y += DYTable_d[ direction ];
        r0.z += DZTable_d[ direction ];
	met.modifyPacking().bitPackedUnset( dpLatticeTmp, met.getCurve().linearizeBoxVectorIndex( r0.x, r0.y, r0.z ) );
        if ( properties & T_Flags(16))
            dpPolymerSystem[ iMonomer ] = r0;
    }
}


/**
 * find jumps and "deapply" them. We just have to find jumps larger than
 * the number of time steps calculated assuming the monomers can only move
 * 1 cell per time step per direction (meaning this also works with
 * diagonal moves!)
 * If for example the box size is 32, but we also calculate with uint8,
 * then the particles may seemingly move not only bei +-32, but also by
 * +-256, but in both cases the particle actually only moves one virtual
 * box.
 * E.g. the particle was at 0 moved to -1 which was mapped to 255 because
 * uint8 overflowed, but the box size is 64, then deltaMove=255 and we
 * need to subtract 3*64. This only works if the box size is a multiple of
 * the type maximum number (256). I.e. in any sane environment if the box
 * size is a power of two, which was a requirement anyway already.
 * Actually, as the position is just calculated as +-1 without any wrapping,
 * the only way for jumps to happen is because of type overflows.
 */
template< typename T_UCoordinateCuda >
__global__ void kernelTreatOverflows
(
    typename CudaVec4< T_UCoordinateCuda >::value_type
                  * const dpPolymerSystemOld        ,
    typename CudaVec4< T_UCoordinateCuda >::value_type
                  * const dpPolymerSystem           ,
    T_Coordinates * const dpiPolymerSystemVirtualBox,
    T_Id            const nMonomers
)
{
    for ( auto iMonomer = blockIdx.x * blockDim.x + threadIdx.x;
          iMonomer < nMonomers; iMonomer += gridDim.x * blockDim.x )
    {
        auto const r0 = dpPolymerSystemOld        [ iMonomer ];
        auto       r1 = dpPolymerSystem           [ iMonomer ];
        auto       iv = dpiPolymerSystemVirtualBox[ iMonomer ];
        T_Coordinates const dr = {
            T_Coordinate( r1.x ) - T_Coordinate( r0.x ),
            T_Coordinate( r1.y ) - T_Coordinate( r0.y ),
            T_Coordinate( r1.z ) - T_Coordinate( r0.z )
        };

        auto constexpr boxSizeCudaType = 1ll << ( sizeof( T_UCoordinateCuda ) * CHAR_BIT );
        assert( boxSizeCudaType >= dcBoxX );
        assert( boxSizeCudaType >= dcBoxY );
        assert( boxSizeCudaType >= dcBoxZ );
        //assert( nMonteCarloSteps < boxSizeCudaType / 2 );
        //assert( nMonteCarloSteps <= std::min( std::min( mBoxX, mBoxY ), mBoxZ ) / 2 );

        if ( std::abs( dr.x ) > T_UCoordinateCuda( boxSizeCudaType / 2 ) )
        {
            r1.x -= boxSizeCudaType - dcBoxX;
            iv.x -= dr.x > decltype( dr.x )(0) ? 1 : -1;
        }
        if ( std::abs( dr.y ) > T_UCoordinateCuda( boxSizeCudaType / 2 ) )
        {
            r1.y -= boxSizeCudaType - dcBoxY;
            iv.y -= dr.y > decltype( dr.y )(0) ? 1 : -1;
        }
        if ( std::abs( dr.z ) > T_UCoordinateCuda( boxSizeCudaType / 2 ) )
        {
            r1.z -= boxSizeCudaType - dcBoxZ;
            iv.z -= dr.z > decltype( dr.z )(0) ? 1 : -1;
        }

        dpPolymerSystem           [ iMonomer ] = r1;
        dpiPolymerSystemVirtualBox[ iMonomer ] = iv;
    }
}

} // end anonymous namespace with typedefs for kernels


template< typename T_UCoordinateCuda >
UpdaterGPUScBFM< T_UCoordinateCuda >::UpdaterGPUScBFM()
 : mStream                          ( 0    ),
   mAge                             ( 0    ),
   mIsPeriodicX                     ( true ),
   mIsPeriodicY                     ( true ),
   mIsPeriodicZ                     ( true ),
   mUsePeriodicMonomerSorting       ( true ),
   mnStepsBetweenSortings           ( 5000 ),
   mLatticeOut                      ( NULL ),
   mLatticeTmp                      ( NULL ),
   mLatticeTmp2                     ( NULL ),
   mnAllMonomers                    ( 0    ),
   mnMonomersPadded                 ( 0    ),
   mPolymerSystem                   ( NULL ),
   mPolymerSystemSorted             ( NULL ),
   mPolymerSystemSortedOld          ( NULL ),
   mviPolymerSystemSortedVirtualBox ( NULL ),
   mPolymerFlags                    ( NULL ),
   miToiNew                         ( NULL ),
   miNewToi                         ( NULL ),
   miNewToiComposition              ( NULL ),
   miNewToiSpatial                  ( NULL ),
   mvKeysZOrderLinearIds            ( NULL ),
   mNeighbors                       ( NULL ),
   mNeighborsSorted                 ( NULL ),
   mNeighborsSortedSizes            ( NULL ),
   mNeighborsSortedInfo             ( nBytesAlignment ),
   mBoxX                            ( 0    ),
   mBoxY                            ( 0    ),
   mBoxZ                            ( 0    ),
   mBoxXM1                          ( 0    ),
   mBoxYM1                          ( 0    ),
   mBoxZM1                          ( 0    ),
   mBoxXLog2                        ( 0    ),
   mBoxXYLog2                       ( 0    ),
   mnSplitColors                    ( 0    ),
   mGlobalIterator                  ( 0    ),
   bSetAutoColoring                 ( true )
{
    /**
     * Log control.
     * Note that "Check" controls not the output, but the actualy checks
     * If a checks needs to always be done, then do that check and declare
     * the output as "Info" log level
     */
    mLog.file( __FILENAME__ );
    mLog.deactivate( "Check"     );
    mLog.deactivate( "Error"     );
    mLog.activate( "Info"      );
    mLog.deactivate( "Stats"     );
    mLog.deactivate( "Warning"   );
}


/**
 * Deletes everything which could and is allocated
 */
template< typename T_UCoordinateCuda >
void UpdaterGPUScBFM< T_UCoordinateCuda >::destruct()
{
    DeleteMirroredObject deletePointer;
    deletePointer( mLatticeOut                     , "mLatticeOut"                      );
    deletePointer( mLatticeTmp                     , "mLatticeTmp"                      );
    deletePointer( mLatticeTmp2                    , "mLatticeTmp2"                     );
    deletePointer( mPolymerSystem                  , "mPolymerSystem"                   );
    deletePointer( mPolymerSystemSorted            , "mPolymerSystemSorted"             );
    deletePointer( mPolymerSystemSortedOld         , "mPolymerSystemSortedOld"          );
    deletePointer( mviPolymerSystemSortedVirtualBox, "mviPolymerSystemSortedVirtualBox" );
    deletePointer( mPolymerFlags                   , "mPolymerFlags"                    );
    deletePointer( miToiNew                        , "miToiNew"                         );
    deletePointer( miNewToi                        , "miNewToi"                         );
    deletePointer( miNewToiComposition             , "miNewToiComposition"              );
    deletePointer( miNewToiSpatial                 , "miNewToiSpatial"                  );
    deletePointer( mvKeysZOrderLinearIds           , "mvKeysZOrderLinearIds"            );
    deletePointer( mNeighbors                      , "mNeighbors"                       );
    deletePointer( mNeighborsSorted                , "mNeighborsSorted"                 );
    deletePointer( mNeighborsSortedSizes           , "mNeighborsSortedSizes"            );
    if ( deletePointer.nBytesFreed > 0 )
    {
        mLog( "Info" )
            << "Freed a total of "
            << prettyPrintBytes( deletePointer.nBytesFreed )
            << " on GPU and host RAM.\n";
    }
}

template< typename T_UCoordinateCuda >
UpdaterGPUScBFM< T_UCoordinateCuda >::~UpdaterGPUScBFM(){ this->destruct(); }

template< typename T_UCoordinateCuda >
void UpdaterGPUScBFM< T_UCoordinateCuda >::setGpu( int iGpuToUse )
{
    int nGpus;
    getCudaDeviceProperties( NULL, &nGpus, true /* print GPU information */ );
    if ( ! ( 0 <= iGpuToUse && iGpuToUse < nGpus ) )
    {
        std::stringstream msg;
        msg << "[" << __FILENAME__ << "::setGpu] "
            << "GPU with ID " << iGpuToUse << " not present. "
            << "Only " << nGpus << " GPUs are available.\n";
        mLog( "Error" ) << msg.str();
        throw std::invalid_argument( msg.str() );
    }
    CUDA_ERROR( cudaSetDevice( iGpuToUse ) );
    miGpuToUse = iGpuToUse;
}

template< typename T_UCoordinateCuda >
void UpdaterGPUScBFM< T_UCoordinateCuda >::initializeBondTable( void )
{
    checkBondVector.initBondTable();

    /* create a table mapping the random int to directions whereto move the
     * monomers. We can use negative numbers, because (0u-1u)+1u still is 0u */
    uint32_t tmp_DXTable[6] = { 0u-1u,1,  0,0,  0,0 };
    uint32_t tmp_DYTable[6] = {  0,0, 0u-1u,1,  0,0 };
    uint32_t tmp_DZTable[6] = {  0,0,  0,0, 0u-1u,1 };
    CUDA_ERROR( cudaMemcpyToSymbol( DXTable_d, tmp_DXTable, sizeof( tmp_DXTable ) ) );
    CUDA_ERROR( cudaMemcpyToSymbol( DYTable_d, tmp_DYTable, sizeof( tmp_DXTable ) ) );
    CUDA_ERROR( cudaMemcpyToSymbol( DZTable_d, tmp_DZTable, sizeof( tmp_DXTable ) ) );
    uint32_t tmp_DXTable2[6] = { 0u-2u,2,  0,0,  0,0 };
    uint32_t tmp_DYTable2[6] = {  0,0, 0u-2u,2,  0,0 };
    uint32_t tmp_DZTable2[6] = {  0,0,  0,0, 0u-2u,2 };
    CUDA_ERROR( cudaMemcpyToSymbol( DXTable2_d, tmp_DXTable2, sizeof( tmp_DXTable2 ) ) );
    CUDA_ERROR( cudaMemcpyToSymbol( DYTable2_d, tmp_DYTable2, sizeof( tmp_DXTable2 ) ) );
    CUDA_ERROR( cudaMemcpyToSymbol( DZTable2_d, tmp_DZTable2, sizeof( tmp_DXTable2 ) ) );
}

template< typename T_UCoordinateCuda >
void UpdaterGPUScBFM< T_UCoordinateCuda >::setAutoColoring( bool bSetAutoColoring_){bSetAutoColoring=bSetAutoColoring_;}

template< typename T_UCoordinateCuda >
void UpdaterGPUScBFM< T_UCoordinateCuda >::initializeSpeciesSorting( void )
{
    if (bSetAutoColoring){
      mLog( "Info" ) << "Coloring graph ...\n";
      bool const bUniformColors = true; // setting this to true should yield more performance as the kernels are uniformly utilized
      mGroupIds = graphColoring< MonomerEdges const *, T_Id, T_Color >(
	  mNeighbors->host, mNeighbors->nElements, bUniformColors,
	  []( MonomerEdges const * const & x, T_Id const & i ){ return x[i].size; },
	  []( MonomerEdges const * const & x, T_Id const & i, size_t const & j ){ return x[i].neighborIds[j]; }
      );
      // mGroupIds is std::vector< T_Color >, i.e., std::vector< uint32_t >

      /* split colors for the parallel -> serial transition tests */
      if ( mnSplitColors > 0 )
      {
	  /* count colors */
	  std::map< T_Color, bool > usedColors;
	  for ( auto const & x : mGroupIds )
	      usedColors[ x ] = true;
	  auto nColors = usedColors.size();
	  /* assert that colors were given from 0 increasing */
	  for ( auto i = 0u; i < nColors; ++i )
	  {
	      if ( !( usedColors.find( i ) != usedColors.end() ) )
		  throw std::runtime_error( "The color splitting algorithm does not work if the initial colors are anything but 0,1,2,...,nColors-1" );
	  }
	  std::map< T_Color, bool > colorWasChanged;
	  for ( auto i = 0u; i < mnSplitColors; ++i )
	  {
	      #if 1
	      /* split in an interleaved fashion like ABCDABCDABCDABCD */
	      for ( auto iColor = 0u; iColor < nColors; ++iColor )
		  colorWasChanged[ iColor ] = true;
	      for ( auto & x : mGroupIds )
	      {
		  /* color was not changed the last time we encountered a monomer of that color, so we have to change the color now */
		  if ( ! colorWasChanged[ x ] )
		  {
		      colorWasChanged[ x ] = true;
		      x += nColors;
		  }
		  else
		      colorWasChanged[ x ] = false;
	      }
	      #else
	      /* split like subchains AAAABBBBCCCCDDDD */
	      /* dirty fix which only works because the BFM file sorts the data already by chain, i.e., chain 0 is monomer [0,511], chain 1 is [512,1023] and so on */

	      #if 1
	      std::vector< uint32_t > nChanged( nColors, 0 );
	      for ( auto & x : mGroupIds )
		  if ( nChanged.at( x )++ < mnAllMonomers / ( nColors * 2 ) )
		      x += nColors;
	      #else // unfinished better color splitting algo
	      std::vector< bool > visited( nMonomers, false );
	      for ( auto iMonomer = 0; iMonomer < mnAllMonomers; ++iMonomer )
	      {
		  if ( visited[ iMonomer ] )
		      continue;
		  visited[ iMonomer ] = true;
		  iChainMonomer = iMonomer;

		  /* iterate to one end of the chain */
		  auto const iStartMonomer = iChainMonomer;
		  while ( mNeighbors->host[ iChainMonomer ].size > 0 )
		  {
		      iChainMonomer = mNeighbors->host[ iChainMonomer ].size;
		      if ( iChainMonomer == iStartMonomer )
			  break;
		  }
		  auto const nNeighbors = ;

		      if ( nNeighbors <= 1 )
			  viEndMonomers.push_back( iChainMonomer );

		      /* Push unvisited monomer to the todo list */
		      for ( auto iBond = 0u; iBond < nNeighbors; ++iBond )
		      {
			  auto const iNeighbor = molecules.getNeighborIdx( iChainMonomer, iBond );
			  if ( ! visited[ iNeighbor ] )
			  {
			      ++nMonomersPerChain;
			      visited[ iNeighbor ] = true;
			      vToDo.push( iNeighbor );
			  }
		      } // loop over neighbors per chain monomer
		  } // loop over monomers inside a chain
	      } // loop over all monomers
	      #endif

	      #endif
	      nColors *= 2;
	  }
      }
    
      /* check automatic coloring with that given in BFM-file */
      if ( mLog.isActive( "Check" ) )
      {
	  mLog( "Info" ) << "Checking difference between automatic and given coloring ... ";
	  size_t nDifferent = 0;
	  for ( size_t iMonomer = 0u; iMonomer < std::max< size_t >( 20, mnAllMonomers ); ++iMonomer )
	  {
	      if ( int32_t( mGroupIds.at( iMonomer )+1 ) != mAttributeSystem[ iMonomer ] )
	      {
		  mLog( "Info" ) << "Color of " << iMonomer << " is automatically " << mGroupIds.at( iMonomer )+1 << " vs. " << mAttributeSystem[ iMonomer ] << "\n";
		  ++nDifferent;
	      }
	  }
	  if ( nDifferent > 0 )
	  {
	      std::stringstream msg;
	      msg << "Automatic coloring failed to produce same result as the given one!";
	      mLog( "Error" ) << msg.str();
	      throw std::runtime_error( msg.str() );
	  }
	  mLog( "Info" ) << "OK\n";
      }
    }
    /* count monomers per species before allocating per species arrays and
     * sorting the monomers into them */
    mLog( "Info" ) << "Attributes of first monomers: ";
    mnElementsInGroup.resize(0);
    for ( size_t i = 0u; i < mGroupIds.size(); ++i )
    {
        if ( i < 40 )
            mLog( "Info" ) << char( 'A' + (char) mGroupIds[i] );
        if ( mGroupIds[i] >= mnElementsInGroup.size() )
            mnElementsInGroup.resize( mGroupIds[i]+1, 0 );
        ++mnElementsInGroup[ mGroupIds[i] ];
    }
    mLog( "Info" ) << "\n";
    if ( mLog.isActive( "Stats" ) && mnElementsInGroup.size() <= 8 )
    {
        mLog( "Stats" ) << "Found " << mnElementsInGroup.size() << " groups with the frequencies: ";
        for ( size_t i = 0u; i < mnElementsInGroup.size(); ++i )
        {
            mLog( "Stats" ) << char( 'A' + (char) i ) << ": " << mnElementsInGroup[i] << "x (" << (float) mnElementsInGroup[i] / mnAllMonomers * 100.f << "%), ";
        }
        mLog( "Stats" ) << "\n";
    }

    /**
     * Generate new array which contains all sorted monomers aligned
     * @verbatim
     * ABABABABABA
     * A A A A A A
     *  B B B B B
     * AAAAAA  BBBBB
     *        ^ alignment
     * @endverbatim
     * in the worst case we are only one element ( 4*intCUDA ) over the
     * alignment with each group and need to fill up to nBytesAlignment for
     * all of them */
    /* virtual number of monomers which includes the additional alignment padding */
    mnMonomersPadded = mnAllMonomers + ( nElementsAlignment - 1u ) * mnElementsInGroup.size();

    assert( miToiNew      == NULL );
    assert( miNewToi      == NULL );
    assert( mPolymerFlags == NULL );
    miToiNew      = new MirroredVector< T_Id    >( mnAllMonomers, mStream );
    miNewToi      = new MirroredVector< T_Id    >( mnMonomersPadded, mStream );
    mPolymerFlags = new MirroredVector< T_Flags >( mnMonomersPadded, mStream );
    assert( miToiNew      != NULL );
    assert( miNewToi      != NULL );
    assert( mPolymerFlags != NULL );
    mPolymerFlags->memsetAsync(0); // can do async as it is next needed in runSimulationOnGPU

    /* calculate offsets to each aligned subgroup vector */
    mviSubGroupOffsets.resize( mnElementsInGroup.size() );
    mviSubGroupOffsets.at(0) = 0;
    for ( size_t i = 1u; i < mnElementsInGroup.size(); ++i )
    {
        mviSubGroupOffsets[i] = mviSubGroupOffsets[i-1] +
        ceilDiv( mnElementsInGroup[i-1], nElementsAlignment ) * nElementsAlignment;
        assert( mviSubGroupOffsets[i] - mviSubGroupOffsets[i-1] >= mnElementsInGroup[i-1] );
    }

    /* virtually sort groups into new array and save index mappings */
    auto iSubGroup = mviSubGroupOffsets;   /* stores the next free index for each subgroup */
    for ( size_t i = 0u; i < mnAllMonomers; ++i )
        miToiNew->host[i] = iSubGroup[ mGroupIds[i] ]++;

    /* create convenience reverse mapping */
    std::fill( miNewToi->host, miNewToi->host + miNewToi->nElements, UINT32_MAX );
    for ( size_t iOld = 0u; iOld < mnAllMonomers; ++iOld )
        miNewToi->host[ miToiNew->host[ iOld ] ] = iOld;

    if ( mLog.isActive( "Info" ) )
    {
        mLog( "Info" ) << "mviSubGroupOffsets = { ";
        for ( auto const & x : mviSubGroupOffsets )
            mLog( "Info" ) << x << ", ";
        mLog( "Info" ) << "}\n";

        mLog( "Info" ) << "iSubGroup = { ";
        for ( auto const & x : iSubGroup )
            mLog( "Info" ) << x << ", ";
        mLog( "Info" ) << "}\n";

        mLog( "Info" ) << "mnElementsInGroup = { ";
        for ( auto const & x : mnElementsInGroup )
            mLog( "Info" ) << x << ", ";
        mLog( "Info" ) << "}\n";
    }

    if ( met.isONGPUForOverhead() &&  mUsePeriodicMonomerSorting )
    {
        miNewToi->pushAsync();
        miToiNew->pushAsync();
    }
}

/**
 * Calculates mapping BtoA from AtoB mapping
 */
__global__ void kernelInvertMapping
(
    T_Id         const * const dpiNewToi      ,
    T_Id               * const dpiToiNew      ,
    T_Id                 const nMonomersPadded
)
{
    for ( auto iNew = blockIdx.x * blockDim.x + threadIdx.x;
          iNew < nMonomersPadded; iNew += gridDim.x * blockDim.x )
    {
        auto const iOld = dpiNewToi[ iNew ];
        if ( iOld != UINT32_MAX )
            dpiToiNew[ iOld ] = iNew;
    }
}

/**
 * needs to be called for each species
 */
__global__ void kernelApplyMappingToNeighbors
(
    MonomerEdges const * const dpNeighbors            ,
    T_Id         const * const dpiNewToi              ,
    T_Id         const * const dpiToiNew              ,
    T_Id               * const dpNeighborsSorted      ,
    uint32_t             const rNeighborsPitchElements,
    uint8_t            * const dpNeighborsSortedSizes ,
    T_Id                 const nMonomers
)
{
    /* apply sorting for polymers, see initializePolymerSystemSorted */

    /* apply sorting for neighbor info, see initializeSortedNeighbors */
    for ( auto iMonomer = blockIdx.x * blockDim.x + threadIdx.x;
          iMonomer < nMonomers; iMonomer += gridDim.x * blockDim.x )
    {
        auto const iOld = dpiNewToi[ iMonomer ];
        auto const nNeighbors = dpNeighbors[ iOld ].size;
        dpNeighborsSortedSizes[ iMonomer ] = nNeighbors;
        for ( size_t j = 0u; j < nNeighbors; ++j )
        {
            dpNeighborsSorted[ j * rNeighborsPitchElements + iMonomer ] =
                dpiToiNew[ dpNeighbors[ iOld ].neighborIds[ j ] ];
        }
    }
}

template< typename T_UCoordinateCuda >
__global__ void kernelUndoPolymerSystemSorting
(
    typename CudaVec4< T_UCoordinateCuda >::value_type
                       const * const dpPolymerSystemSorted           ,
    T_Coordinates      const * const dpiPolymerSystemSortedVirtualBox,
    T_Id               const * const dpiNewToi                       ,
    T_Coordinates            * const dpPolymerSystem                 ,
    T_Id                       const nMonomersPadded
)
{
    for ( auto iNew = blockIdx.x * blockDim.x + threadIdx.x;
          iNew < nMonomersPadded; iNew += gridDim.x * blockDim.x )
    {
        auto const iOld = dpiNewToi[ iNew ];
        if ( iOld == UINT32_MAX )
            continue;
        auto const rsmall = dpPolymerSystemSorted[ iNew ];
        /* problematic typecast happens here with T_UCoordinateCuda = uint32_t
         * because it would then get >down<converted to int32_t!
         * But this is ok, because LeMonADE also has this limitation */
        T_Coordinates rSorted = {
            T_Coordinate( rsmall.x ),
            T_Coordinate( rsmall.y ),
            T_Coordinate( rsmall.z ),
            T_Coordinate( rsmall.w )
        };
        auto const nPos = dpiPolymerSystemSortedVirtualBox[ iNew ];
        rSorted.x += nPos.x * dcBoxX;
        rSorted.y += nPos.y * dcBoxY;
        rSorted.z += nPos.z * dcBoxZ;
        dpPolymerSystem[ iOld ] = rSorted;
    }
}

template< typename T_UCoordinateCuda >
__global__ void kernelSplitMonomerPositions
(
    T_Coordinates const * const dpPolymerSystem                 ,
    T_Id          const * const dpiNewToi                       ,
    T_Coordinates       * const dpiPolymerSystemSortedVirtualBox,
    typename CudaVec4< T_UCoordinateCuda >::value_type
                        * const dpPolymerSystemSorted           ,
    size_t                const nMonomersPadded
)
{
    for ( auto iNew = blockIdx.x * blockDim.x + threadIdx.x;
          iNew < nMonomersPadded; iNew += gridDim.x * blockDim.x )
    {
        auto const iOld = dpiNewToi[ iNew ];
        if ( iOld == UINT32_MAX )
            continue;
        auto const r = dpPolymerSystem[ iOld ];
        typename CudaVec4< T_UCoordinateCuda >::value_type rlo = {
            T_UCoordinateCuda( r.x & dcBoxXM1 ),
            T_UCoordinateCuda( r.y & dcBoxYM1 ),
            T_UCoordinateCuda( r.z & dcBoxZM1 ),
            T_UCoordinateCuda( dpPolymerSystemSorted[ iNew ].w )
        };
        dpPolymerSystemSorted[ iNew ] = rlo;
        T_Coordinates rhi = {
            ( r.x - T_Coordinate( rlo.x ) ) / T_Coordinate( dcBoxX ),
            ( r.y - T_Coordinate( rlo.y ) ) / T_Coordinate( dcBoxY ),
            ( r.z - T_Coordinate( rlo.z ) ) / T_Coordinate( dcBoxZ ),
            0
        };
        dpiPolymerSystemSortedVirtualBox[ iNew ] = rhi;
    }
}


template< typename T_UCoordinateCuda >
struct LinearizeBoxVectorIndexFunctor
{
  Method met;
  LinearizeBoxVectorIndexFunctor(const Method& met_ ):met(met_){} 
  using T_UCoordinatesCuda = typename CudaVec4< T_UCoordinateCuda >::value_type;
  __device__ inline T_Id operator()( T_UCoordinatesCuda const & r ) const
  {
      return met.getCurve().linearizeBoxVectorIndex( r.x, r.y, r.z );
  }
};

/**
 * this works on mPolymerSystemSorted and resorts the monomers along a
 * z-order curve in order to improve cache hit rates, especially for "slow"
 * systems. Also it updates the order of and the IDs inside mNeighborsSorted
 * @param[in]  polymerSystemSorted
 * @param[in]  iToiNew specifies the mapping used to create polymerSystemSorted
 *             from polymerSystem, i.e.
 *             polymerSystemSorted[ iToiNew[i] ] == polymerSystem[i]
 * @param[in]  iNewToi same as iToiNew, but the other way arount, i.e.
 *             polymerSystemSorted[i] == polymerSystem[ iNewToi ]
 *             Note that the sorted system includes padding, therefore some
 *             entries of iNewToi contain UINT32_MAX to indicate that those
 *             are not do be mapped
 * @param[out] iToiNew just as the input, but after sorting spatially
 * @param[out] iNewToi
 */
template< typename T_UCoordinateCuda >
void UpdaterGPUScBFM< T_UCoordinateCuda >::doSpatialSorting( void )
{
    auto const nThreads = 128;
    auto const nBlocksP = ceilDiv( mnMonomersPadded, nThreads );
    /* because resorting changes the order we have to do the full
     * overflow checks and also update mPolymerSystemSortedOld ! */
    if ( useOverflowChecks )
    {
        /* the padding values do not change, so we can simply let the threads
         * calculate them without worries and save the loop over the species */
        kernelTreatOverflows< T_UCoordinateCuda >
        <<< nBlocksP, nThreads, 0, mStream >>>(
            mPolymerSystemSortedOld         ->gpu,
            mPolymerSystemSorted            ->gpu,
            mviPolymerSystemSortedVirtualBox->gpu,
            mnMonomersPadded
        );
    }

    /* dependent on kernelTreatOverflows ( @todo if it is dependent, what happens if useOverflowChecks == false !? Shouldn't this also be inside the body of the if-statement ??? Possibly not, if the virtutalBox ID is always zero for no overflow checks being used :S ) */
    kernelUndoPolymerSystemSorting< T_UCoordinateCuda >
    <<< nBlocksP, nThreads, 0, mStream >>>
    (
        mPolymerSystemSorted            ->gpu,
        mviPolymerSystemSortedVirtualBox->gpu,
        miNewToi                        ->gpu,
        mPolymerSystem                  ->gpu,
        mnMonomersPadded
    );

    /* mapping new (monomers spatially sorted) index to old (species sorted) index */
    if ( miNewToiComposition   == NULL ) miNewToiComposition   = new MirroredVector< T_Id >( mnMonomersPadded, mStream );
    if ( miNewToiSpatial       == NULL ) miNewToiSpatial       = new MirroredVector< T_Id >( mnMonomersPadded, mStream );
    if ( mvKeysZOrderLinearIds == NULL ) mvKeysZOrderLinearIds = new MirroredVector< T_Id >( mnMonomersPadded, mStream );
    assert( miNewToiComposition   != NULL );
    assert( miNewToiSpatial       != NULL );
    assert( mvKeysZOrderLinearIds != NULL );

    /* @see https://thrust.github.io/doc/group__transformations.html#ga233a3db0c5031023c8e9385acd4b9759
       @see https://thrust.github.io/doc/group__transformations.html#ga281b2e453bfa53807eda1d71614fb504 */
    /* not dependent on anything, could run in different stream */
    thrust::sequence( thrust::system::cuda::par, miNewToiSpatial->gpu, miNewToiSpatial->gpu + miNewToiSpatial->nElements );
    /* dependent on above, but does not depend on kernelUndoPolymerSystemSorting */
    thrust::transform( thrust::system::cuda::par,
        mPolymerSystemSorted ->gpu,
        mPolymerSystemSorted ->gpu + mPolymerSystemSorted->nElements,
        mvKeysZOrderLinearIds->gpu,
        LinearizeBoxVectorIndexFunctor< T_UCoordinateCuda >(met)
    );
    /* sort per sublists (each species) by key, not the whole list */
    for ( auto iSpecies = 0u; iSpecies < mnElementsInGroup.size(); ++iSpecies )
    {
        thrust::sort_by_key( thrust::system::cuda::par,
            mvKeysZOrderLinearIds->gpu + mviSubGroupOffsets.at( iSpecies ),
            mvKeysZOrderLinearIds->gpu + mviSubGroupOffsets.at( iSpecies ) + mnElementsInGroup.at( iSpecies ),
            miNewToiSpatial      ->gpu + mviSubGroupOffsets.at( iSpecies )
        );
    }

    thrust::fill( thrust::system::cuda::par, miNewToiComposition->gpu, miNewToiComposition->gpu + miNewToiComposition->nElements, UINT32_MAX );
    /**
     * @see https://thrust.github.io/doc/group__gathering.html#ga86722e76264fb600d659c1adef5d51b2
     *   -> for ( it : map ) result[ it - map_first ] = input_first[ *it ]
     *   -> for ( i ) result[i] = input_first[ map[i] ]
     * for ( T_Id iNew = 0u; iNew < miNewToiSpatial->nElements ; ++iNew )
     *     iNewToiComposition.at( iNew ) = miNewToi->host[ miNewToiSpatial->host[ iNew ] ];
     */
    thrust::gather( thrust::system::cuda::par,
        miNewToiSpatial    ->gpu,
        miNewToiSpatial    ->gpu + miNewToiSpatial->nElements,
        miNewToi           ->gpu,
        miNewToiComposition->gpu
    );
    std::swap( miNewToi->gpu, miNewToiComposition->gpu ); // avoiding memcpy by swapping pointers on GPU
    kernelInvertMapping<<< nBlocksP, nThreads >>>( miNewToi->gpu, miToiNew->gpu, miNewToi->nElements );
    for ( auto iSpecies = 0u; iSpecies < mnElementsInGroup.size(); ++iSpecies )
    {
        kernelApplyMappingToNeighbors<<< nBlocksP, nThreads, 0, mStream >>>(
            mNeighbors           ->gpu,
            miNewToi             ->gpu + mviSubGroupOffsets[ iSpecies ],
            miToiNew             ->gpu,
            mNeighborsSorted     ->gpu + mNeighborsSortedInfo.getMatrixOffsetElements( iSpecies ),
            mNeighborsSortedInfo.getMatrixPitchElements( iSpecies ),
            mNeighborsSortedSizes->gpu + mviSubGroupOffsets[ iSpecies ],
            mnElementsInGroup[ iSpecies ]
        );
    }

    /* kernelUndoPolymerSystemSorting followed by kernelSplitMonomerPositions
     * basically just avoids using two temporary arrays for the resorting of
     * mPolymerSystemSorted and mviPolymerSystemSortedVirtualBox */

    /* dependent on:
     *   kernelUndoPolymerSystemSorting (mPolymerSystem)
     *   thrust::transform (mPolymerSystemSorted)
     *   thrust::gather (miNewToi)
     */
    kernelSplitMonomerPositions< T_UCoordinateCuda >
    <<< nBlocksP, nThreads, 0, mStream >>>(
        mPolymerSystem                  ->gpu,
        miNewToi                        ->gpu,
        mviPolymerSystemSortedVirtualBox->gpu,
        mPolymerSystemSorted            ->gpu,
        mnMonomersPadded
    );

    CUDA_ERROR( cudaMemcpyAsync( mPolymerSystemSortedOld->gpu, mPolymerSystemSorted->gpu, mPolymerSystemSortedOld->nBytes, cudaMemcpyDeviceToDevice, mStream ) );
}

template< typename T_UCoordinateCuda >
void UpdaterGPUScBFM< T_UCoordinateCuda >::initializeSortedNeighbors( void )
{
    /* adjust neighbor IDs to new sorted PolymerSystem and also sort that array.
     * Bonds are not supposed to change, therefore we don't need to push and
     * pop them each time we do something on the GPU! */

    assert( mNeighborsSortedInfo.getRequiredBytes() == 0 );
    for ( size_t i = 0u; i < mnElementsInGroup.size(); ++i )
        mNeighborsSortedInfo.newMatrix( MAX_CONNECTIVITY, mnElementsInGroup[i] );
    if ( mNeighborsSorted      == NULL ) mNeighborsSorted      = new MirroredVector< T_Id    >( mNeighborsSortedInfo.getRequiredElements(), mStream );
    if ( mNeighborsSortedSizes == NULL ) mNeighborsSortedSizes = new MirroredVector< uint8_t >( mnMonomersPadded, mStream );
    assert( mNeighborsSorted      != NULL );
    assert( mNeighborsSortedSizes != NULL );

    if ( mLog.isActive( "Info" ) )
    {
        mLog( "Info" )
        << "Allocated mNeighborsSorted with "
        << mNeighborsSorted->nElements << " elements in "
        << mNeighborsSorted->nBytes << "B ("
        << mNeighborsSortedInfo.getRequiredElements() << ","
        << mNeighborsSortedInfo.getRequiredBytes() << "B)\n";

        mLog( "Info" ) << "mNeighborsSortedInfo:\n";
        for ( size_t iSpecies = 0u; iSpecies < mnElementsInGroup.size(); ++iSpecies )
        {
            mLog( "Info" )
            << "== matrix/species " << iSpecies << " ==\n"
            << "offset:" << mNeighborsSortedInfo.getMatrixOffsetElements( iSpecies ) << " elements = "
                         << mNeighborsSortedInfo.getMatrixOffsetBytes   ( iSpecies ) << "B\n"
            //<< "rows  :" << mNeighborsSortedInfo.getOffsetElements() << " elements = "
            //             << mNeighborsSortedInfo.getOffsetBytes() << "B\n"
            //<< "cols  :" << mNeighborsSortedInfo.getOffsetElements() << " elements = "
            //             << mNeighborsSortedInfo.getOffsetBytes() << "B\n"
            << "pitch :" << mNeighborsSortedInfo.getMatrixPitchElements( iSpecies ) << " elements = "
                         << mNeighborsSortedInfo.getMatrixPitchBytes   ( iSpecies ) << "B\n";
        }
        mLog( "Info" ) << "[UpdaterGPUScBFM::initializeSortedNeighbors] map neighborIds to sorted array ... ";
    }

    if (met.isONGPUForOverhead()){
      auto const nThreads = 128;
      auto const nBlocksP = ceilDiv( mnMonomersPadded, nThreads );
      for ( auto iSpecies = 0u; iSpecies < mnElementsInGroup.size(); ++iSpecies )
      {
	  kernelApplyMappingToNeighbors<<< nBlocksP, nThreads, 0, mStream >>>(
	      mNeighbors           ->gpu,
	      miNewToi             ->gpu + mviSubGroupOffsets[ iSpecies ],
	      miToiNew             ->gpu,
	      mNeighborsSorted     ->gpu + mNeighborsSortedInfo.getMatrixOffsetElements( iSpecies ),
	      mNeighborsSortedInfo.getMatrixPitchElements( iSpecies ),
	      mNeighborsSortedSizes->gpu + mviSubGroupOffsets[ iSpecies ],
	      mnElementsInGroup[ iSpecies ]
	  );
      }
    }else{
      for ( auto iSpecies = 0u; iSpecies < mnElementsInGroup.size(); ++iSpecies )
      {
	  for ( size_t iMonomer = 0u; iMonomer < mnElementsInGroup[ iSpecies ]; ++iMonomer )
	  {
	      auto const i = mviSubGroupOffsets[ iSpecies ] + iMonomer;
	      auto const iOld = miNewToi->host[i];

	      mNeighborsSortedSizes->host[i] = mNeighbors->host[ iOld ].size;
	      auto const pitch = mNeighborsSortedInfo.getMatrixPitchElements( iSpecies );
	      for ( size_t j = 0u; j < mNeighbors->host[ iOld ].size; ++j )
	      {
		  if ( i < 5 || std::abs( (long long int) i - mviSubGroupOffsets[ mviSubGroupOffsets.size()-1 ] ) < 5 )
		  {
		      mLog( "Info" ) << "Currently at index " << i << ": Writing into mNeighborsSorted->host[ " << mNeighborsSortedInfo.getMatrixOffsetElements( iSpecies ) << " + " << j << " * " << pitch << " + " << i << "-" << mviSubGroupOffsets[ iSpecies ] << "] the value of old neighbor located at miToiNew->host[ mNeighbors[ miNewToi->host[i]=" << miNewToi->host[i] << " ] = miToiNew->host[ " << mNeighbors->host[ miNewToi->host[i] ].neighborIds[j] << " ] = " << miToiNew->host[ mNeighbors->host[ miNewToi->host[i] ].neighborIds[j] ] << " \n";
		  }
		  auto const iNeighborSorted = mNeighborsSortedInfo.getMatrixOffsetElements( iSpecies )
					    + j * pitch + iMonomer;
		  mNeighborsSorted->host[ iNeighborSorted ] = miToiNew->host[ mNeighbors->host[ iOld ].neighborIds[ j ] ];
	      }
	  }
      }

      mNeighborsSorted     ->pushAsync();
      mNeighborsSortedSizes->pushAsync();
      mLog( "Info" ) << "Done\n";
    }

    /* some checks for correctness of new adjusted neighbor global IDs */
    if ( mLog.isActive( "Check" ) )
    {
        /* note that this also checks "unitialized entries" those should be
         * initialized to 0 to reduce problems. This is done by the memset. */
        /*for ( size_t i = 0u; i < mNeighborsSorted->nElements; ++i )
        {
            if ( mNeighbors[i].size > MAX_CONNECTIVITY )
                throw std::runtime_error( "A monomer has more neighbors than allowed!" );
            for ( size_t j = 0u; j < mNeighbors[i].size; ++j )
            {
                auto const iSorted = mNeighborsSorted->host[i].neighborIds[j];
                if ( iSorted == UINT32_MAX )
                    throw std::runtime_error( "New index mapping not set!" );
                if ( iSorted >= mnMonomersPadded )
                    throw std::runtime_error( "New index out of range!" );
            }
        }*/
        /* does a similar check for the unsorted error which is still used
         * to create the property tag */
        for ( T_Id i = 0; i < mnAllMonomers; ++i )
        {
            if ( mNeighbors->host[i].size > MAX_CONNECTIVITY )
            {
                std::stringstream msg;
                msg << "[" << __FILENAME__ << "::initializeSortedNeighbors] "
                    << "This implementation allows max. 7 neighbors per monomer, "
                    << "but monomer " << i << " has " << mNeighbors->host[i].size << "\n";
                mLog( "Error" ) << msg.str();
                throw std::invalid_argument( msg.str() );
            }
        }
    }
}


template< typename T_UCoordinateCuda >
void UpdaterGPUScBFM< T_UCoordinateCuda >::initializeSortedMonomerPositions( void )
{
    /* sort groups into new array and save index mappings */
    if ( mPolymerSystemSorted             == NULL )mPolymerSystemSorted             = new MirroredVector< T_UCoordinatesCuda >( mnMonomersPadded, mStream );
    if ( mPolymerSystemSortedOld          == NULL )mPolymerSystemSortedOld          = new MirroredVector< T_UCoordinatesCuda >( mnMonomersPadded, mStream );
    if ( mviPolymerSystemSortedVirtualBox == NULL )mviPolymerSystemSortedVirtualBox = new MirroredVector< T_Coordinates      >( mnMonomersPadded, mStream );
    assert( mPolymerSystemSorted             != NULL );
    assert( mPolymerSystemSortedOld          != NULL );
    assert( mviPolymerSystemSortedVirtualBox != NULL );
    #ifndef NDEBUG
        mPolymerSystemSorted            ->memset( 0 );
        mPolymerSystemSortedOld         ->memset( 0 );
        mviPolymerSystemSortedVirtualBox->memset( 0 );
    #endif

    if(met.isONGPUForOverhead()){
      auto const nThreads = 128;
      auto const nBlocksP = ceilDiv( mnMonomersPadded, nThreads );
      kernelSplitMonomerPositions< T_UCoordinateCuda >
      <<< nBlocksP, nThreads, 0, mStream >>>(
	  mPolymerSystem                  ->gpu,
	  miNewToi                        ->gpu,
	  mviPolymerSystemSortedVirtualBox->gpu,
	  mPolymerSystemSorted            ->gpu,
	  mnMonomersPadded
      );
    }else{
      mLog( "Info" ) << "[" << __FILENAME__ << "::initializeSortedMonomerPositions] sort mPolymerSystem -> mPolymerSystemSorted ...\n";
      for ( T_Id i = 0u; i < mnAllMonomers; ++i )
      {
	  if ( i < 20 )
	      mLog( "Info" ) << "Write " << i << " to " << this->miToiNew->host[i] << "\n";

	  auto const x = mPolymerSystem->host[i].x;
	  auto const y = mPolymerSystem->host[i].y;
	  auto const z = mPolymerSystem->host[i].z;

	  mPolymerSystemSorted->host[ miToiNew->host[i] ].x = x & mBoxXM1;
	  mPolymerSystemSorted->host[ miToiNew->host[i] ].y = y & mBoxYM1;
	  mPolymerSystemSorted->host[ miToiNew->host[i] ].z = z & mBoxZM1;
	  mPolymerSystemSorted->host[ miToiNew->host[i] ].w = mNeighbors->host[i].size;

	  mviPolymerSystemSortedVirtualBox->host[ miToiNew->host[i] ].x = ( x - ( x & mBoxXM1 ) ) / mBoxX;
	  mviPolymerSystemSortedVirtualBox->host[ miToiNew->host[i] ].y = ( y - ( y & mBoxYM1 ) ) / mBoxY;
	  mviPolymerSystemSortedVirtualBox->host[ miToiNew->host[i] ].z = ( z - ( z & mBoxZM1 ) ) / mBoxZ;

	  auto const pTarget  = &mPolymerSystemSorted            ->host[ miToiNew->host[i] ];
	  auto const pTarget2 = &mviPolymerSystemSortedVirtualBox->host[ miToiNew->host[i] ];
	  if ( ! ( ( (T_Coordinate) pTarget->x + (T_Coordinate) pTarget2->x * (T_Coordinate) mBoxX == x ) &&
		  ( (T_Coordinate) pTarget->y + (T_Coordinate) pTarget2->y * (T_Coordinate) mBoxY == y ) &&
		  ( (T_Coordinate) pTarget->z + (T_Coordinate) pTarget2->z * (T_Coordinate) mBoxZ == z )
	  ) )
	  {
	      std::stringstream msg;
	      msg << "[" << __FILENAME__ << "::initializeSortedMonomerPositions] "
		  << "Error while trying to compress the globale positions into box size modulo and number of virtual box the monomer resides in. Virtual box number "
		  << "(" << pTarget2->x << "," << pTarget2->y << "," << pTarget2->z << ")"
		  << ", wrapped position: "
		  << "(" << pTarget->x << "," << pTarget->y << "," << pTarget->z << ")"
		  << " => reconstructed global position ("
		  << pTarget->x + pTarget2->x * mBoxX << ","
		  << pTarget->y + pTarget2->y * mBoxY << ","
		  << pTarget->z + pTarget2->z * mBoxZ << ")"
		  << " should be equal to the input position: "
		  << "(" << x << "," << y << "," << z << ")"
		  << std::endl;
	      throw std::runtime_error( msg.str() );
	  }
      }
      mPolymerSystemSorted            ->pushAsync();
      mviPolymerSystemSortedVirtualBox->pushAsync();
    }
}

template< typename T_UCoordinateCuda >
void UpdaterGPUScBFM< T_UCoordinateCuda >::initializeLattices( void )
{
    if ( mLatticeOut != NULL || mLatticeTmp != NULL )
    {
        std::stringstream msg;
        msg << "[" << __FILENAME__ << "::initializeLattices] "
            << "Initialize was already called and may not be called again "
            << "until cleanup was called!";
        mLog( "Error" ) << msg.str();
        throw std::runtime_error( msg.str() );
    }

    size_t nBytesLatticeTmp = mBoxX * mBoxY * mBoxZ / sizeof( T_Lattice );
    
    if (met.getPacking().getBitPackingOn())
        nBytesLatticeTmp /= CHAR_BIT;
    
    if( met.getPacking().getNBufferedTmpLatticeOn())
        nBytesLatticeTmp *= mnLatticeTmpBuffers;
    
    mLatticeOut  = new MirroredTexture< T_Lattice >( mBoxX * mBoxY * mBoxZ, mStream );
    mLatticeTmp  = new MirroredTexture< T_Lattice >( nBytesLatticeTmp     , mStream );
    mLatticeTmp2 = new MirroredTexture< T_Lattice >( mBoxX * mBoxY * mBoxZ, mStream );
    mLatticeTmp ->memsetAsync(0); // async as it is next needed in runSimulationOnGPU
    mLatticeTmp2->memsetAsync(0);
    /* populate latticeOut with monomers from mPolymerSystem */
    std::memset( mLatticeOut->host, 0, mLatticeOut->nBytes );
    for ( T_Id iMonomer = 0; iMonomer < mnAllMonomers; ++iMonomer )
    {
        mLatticeOut->host[ met.getCurve().linearizeBoxVectorIndex(
            mPolymerSystem->host[ iMonomer ].x,
            mPolymerSystem->host[ iMonomer ].y,
            mPolymerSystem->host[ iMonomer ].z
        ) ] = 1;
    }
    mLatticeOut->pushAsync();

    mLog( "Info" )
        << "Filling Rate: " << mnAllMonomers << " "
        << "(=" << mnAllMonomers / 1024 << "*1024+" << mnAllMonomers % 1024 << ") "
        << "particles in a (" << mBoxX << "," << mBoxY << "," << mBoxZ << ") box "
        << "=> " << 100. * mnAllMonomers / ( mBoxX * mBoxY * mBoxZ ) << "%\n"
        << "Note: densest packing is: 25% -> in this case it might be more reasonable to actually iterate over the spaces where particles can move to, keeping track of them instead of iterating over the particles\n";

    
    if( met.getPacking().getNBufferedTmpLatticeOn()){
        /**
         * Addresses must be aligned to 32=2*4*4 byte boundaries
         * @see https://devtalk.nvidia.com/default/topic/975906/cuda-runtime-api-error-74-misaligned-address/?offset=5
         * Currently the code does not bother with padding the tmp lattice
         * buffers assuming that the box is large enough to automatically
         * lead to the correct alignment. This also assumes the box size to be
         * of power 2
         */
        if ( mBoxX * mBoxY * mBoxZ < 32 )
        {
            std::stringstream msg;
            msg << "[" << __FILENAME__ << "::initializeLattices] [Error] The total cells in the box " << mBoxX * mBoxY * mBoxZ << " is smaller than 32. This is not allowed (yet) with met.getPacking().getNBufferedTmpLatticeOn() turned as it would neccessitate additional padding between the buffers. Please undefine met.getPacking().getNBufferedTmpLatticeOn() in the source code or increase the box size!\n";
            mLog( "Error" ) << msg.str();
            throw std::runtime_error( msg.str() );
        }
        /**
         * "CUDA C Programming Guide 5.0", p73 says "Any address of a variable residing in global memory or returned by one of the memory allocation routines from the driver or runtime API is always aligned to at least 256 bytes"
         * @see https://stackoverflow.com/questions/14082964/cuda-alignment-256bytes-seriously
         */
        else if ( mBoxX * mBoxY * mBoxZ < 256 )
        {
            std::stringstream msg;
            msg << "[" << __FILENAME__ << "::initializeLattices] [Warning] The total cells in the box " << mBoxX * mBoxY * mBoxZ << " is smaller than 256. This might lead to performance loss. Try undefining met.getPacking().getNBufferedTmpLatticeOn() in the source code or increase the box size.\n";
            mLog( "Warning" ) << msg.str();
        }

        mvtLatticeTmp.resize( mnLatticeTmpBuffers );
        cudaResourceDesc mResDesc;
        cudaTextureDesc  mTexDesc;
        std::memset( &mResDesc, 0, sizeof( mResDesc ) );
        mResDesc.resType = cudaResourceTypeLinear;
        mResDesc.res.linear.desc.f = cudaChannelFormatKindUnsigned;
        mResDesc.res.linear.desc.x = sizeof( mLatticeTmp->gpu[0] ) * CHAR_BIT; // bits per channel
        std::memset( &mTexDesc, 0, sizeof( mTexDesc ) );
        mTexDesc.readMode = cudaReadModeElementType;
        for ( auto i = 0u; i < mnLatticeTmpBuffers; ++i )
        {
            mResDesc.res.linear.sizeInBytes = mBoxX * mBoxY * mBoxZ * sizeof( mLatticeTmp->gpu[0] );
	    if (met.getPacking().getBitPackingOn())
                mResDesc.res.linear.sizeInBytes /= CHAR_BIT;
            mResDesc.res.linear.devPtr = (uint8_t*) mLatticeTmp->gpu + i * mResDesc.res.linear.sizeInBytes;
            mLog( "Info" )
                << "Bind texture for " << i << "-th temporary lattice buffer "
                << "to mLatticeTmp->gpu + " << ( i * mResDesc.res.linear.sizeInBytes )
                << "\n";
            /* the last three arguments are pointers to constants! */
            cudaCreateTextureObject( &mvtLatticeTmp.at(i), &mResDesc, &mTexDesc, NULL );
        }
    }
}

template< typename T_UCoordinateCuda >
void UpdaterGPUScBFM< T_UCoordinateCuda >::checkMonomerReorderMapping( void )
{
    if ( miToiNew->nElements != mnAllMonomers )
    {
        std::stringstream msg;
        msg << "[" << __FILENAME__ << "::checkMonomerReorderMapping] "
            << "miToiNew must have " << mnAllMonomers << " elements "
            << "(as many as monomers), but it has " << miToiNew->nElements << "!";
        mLog( "Error" ) << msg.str();
        throw std::runtime_error( msg.str() );
    }

    if ( miNewToi->nElements != mnMonomersPadded )
    {
        std::stringstream msg;
        msg << "[" << __FILENAME__ << "::checkMonomerReorderMapping] "
            << "miNewToi must have " << mnMonomersPadded << " elements "
            << "(as many as monomers), but it has " << miNewToi->nElements << "!";
        mLog( "Error" ) << msg.str();
        throw std::runtime_error( msg.str() );
    }

    auto const nMonomers       = miToiNew->nElements;
    auto const nMonomersPadded = miNewToi->nElements;

    /* check that the mapping is bijective if we exclude
     * entries equal to UINT32_MAX */
    std::vector< bool > vIsMapped( nMonomers, false );

    for ( size_t iNew = 0u; iNew < miNewToi->nElements; ++iNew )
    {
        auto const iOld = miNewToi->host[ iNew ];

        if ( iOld == UINT32_MAX )
            continue;

        if ( ! ( /* 0 <= iOld && */ iOld < nMonomers ) )
        {
            std::stringstream msg;
            msg << "[" << __FILENAME__ << "::checkMonomerReorderMapping] "
                << "New index " << iNew << " is mapped back to " << iOld
                << ", which is out of range [0," << nMonomers-1 << "]";
            mLog( "Error" ) << msg.str();
            throw std::runtime_error( msg.str() );
        }

        if ( vIsMapped.at( iOld ) )
        {
            std::stringstream msg;
            msg << "[" << __FILENAME__ << "::checkMonomerReorderMapping] "
                << "When trying to test whether we can copy back the monomer "
                << "at the sorted index " << iNew << ", it was found that the "
                << "index " << iOld << " in the unsorted array was already "
                << "written to, i.e., we would loose information on copy-back!";
            mLog( "Error" ) << msg.str();
            throw std::runtime_error( msg.str() );
        }
        vIsMapped.at( iOld ) = true;
    }

    size_t const nMapped = std::count( vIsMapped.begin(), vIsMapped.end(), true );
    if ( nMapped != nMonomers )
    {
        std::stringstream msg;
        msg << "[" << __FILENAME__ << "::checkMonomerReorderMapping] "
            << "The mapping created e.g. by the sorting by species is missing "
            << "some monomers! Only " << nMapped << " / " << nMonomers
            << " are actually mapped to the new sorted array!";
        mLog( "Error" ) << msg.str();
        throw std::runtime_error( msg.str() );
    }

    std::vector< bool > vIsMappedTo( nMonomersPadded, false );

    for ( size_t iOld = 0u; iOld < miToiNew->nElements; ++iOld )
    {
        auto const iNew = miToiNew->host[ iOld ];

        if ( iNew == UINT32_MAX )
            continue;

        if ( ! ( /* 0 <= iNew && */ iNew < mnMonomersPadded ) )
        {
            std::stringstream msg;
            msg << "[" << __FILENAME__ << "::checkMonomerReorderMapping] "
                << "Old index " << iOld << " maps to " << iNew << ", which is "
                << "out of range [0," << mnMonomersPadded-1 << "]";
            mLog( "Error" ) << msg.str();
            throw std::runtime_error( msg.str() );
        }

        if ( vIsMappedTo.at( iNew ) )
        {
            std::stringstream msg;
            msg << "[" << __FILENAME__ << "::checkMonomerReorderMapping] "
                << "When trying to test whether we can copy back the monomer "
                << "at the sorted index " << iNew << ", it was found that the "
                << "index " << iOld << " in the unsorted array was already "
                << "written to, i.e., we would loose information on copy-back!";
            mLog( "Error" ) << msg.str();
            throw std::runtime_error( msg.str() );
        }
        vIsMappedTo.at( iNew ) = true;
    }

    size_t const nMappedTo = std::count( vIsMappedTo.begin(), vIsMappedTo.end(), true );
    if ( nMappedTo != nMonomers )
    {
        std::stringstream msg;
        msg << "[" << __FILENAME__ << "::checkMonomerReorderMapping] "
            << "The mapping created e.g. by the sorting by species is missing "
            << "some monomers! The sorted array only has " << nMappedTo << " / "
            << nMonomers << " monomers!";
        mLog( "Error" ) << msg.str();
        throw std::runtime_error( msg.str() );
    }

    /* check that it actually is the inverse */
    for ( size_t iOld = 0u; iOld < miToiNew->nElements; ++iOld )
    {
        if ( miToiNew->host[ iOld ] == UINT32_MAX )
            continue;

        if ( miNewToi->host[ miToiNew->host[ iOld ] ] != iOld )
        {
            std::stringstream msg;
            msg << "[" << __FILENAME__ << "::checkMonomerReorderMapping] "
                << "Roundtrip iOld -> iNew -> iOld not working for iOld= "
                << iOld << " -> " << miToiNew->host[ iOld ] << " -> "
                << miNewToi->host[ miToiNew->host[ iOld ] ];
            mLog( "Error" ) << msg.str();
            throw std::runtime_error( msg.str() );
        }
    }

    /* check that it actually is the inverse the other way around */
    for ( size_t iNew = 0u; iNew < miNewToi->nElements; ++iNew )
    {
        if ( miNewToi->host[ iNew ] == UINT32_MAX )
            continue;

        if ( miToiNew->host[ miNewToi->host[ iNew ] ] != iNew )
        {
            std::stringstream msg;
            msg << "[" << __FILENAME__ << "::checkMonomerReorderMapping] "
                << "Roundtrip iNew -> iOld -> iNew not working for iNew= "
                << iNew << " -> " << miNewToi->host[ iNew ] << " -> "
                << miToiNew->host[ miNewToi->host[ iNew ] ];
            mLog( "Error" ) << msg.str();
            throw std::runtime_error( msg.str() );
        }
    }
}

template< typename T_UCoordinateCuda >
void UpdaterGPUScBFM< T_UCoordinateCuda >::initialize( void )
{
    if ( mLog( "Stats" ).isActive() )
    {
        // this is called in parallel it seems, therefore need to buffer it
        std::stringstream msg; msg
        << "[" << __FILENAME__ << "::initialize] The "
        << "(" << mBoxX << "," << mBoxY << "," << mBoxZ << ")"
        << " lattice is populated by " << mnAllMonomers
        << " resulting in a filling rate of "
        << mnAllMonomers / double( mBoxX * mBoxY * mBoxZ ) << "\n";
        mLog( "Stats" ) << msg.str();
    }

    mLog( "Info" )
    << "T_BoxSize          = " << getTypeInfoString< T_BoxSize          >() << "\n"
    << "T_Coordinate       = " << getTypeInfoString< T_Coordinate       >() << "\n"
    << "T_CoordinateCuda   = " << getTypeInfoString< T_CoordinateCuda   >() << "\n"
    << "T_UCoordinateCuda  = " << getTypeInfoString< T_UCoordinateCuda  >() << "\n"
    << "T_Coordinates      = " << getTypeInfoString< T_Coordinates      >() << "\n"
    << "T_CoordinatesCuda  = " << getTypeInfoString< T_CoordinatesCuda  >() << "\n"
    << "T_UCoordinatesCuda = " << getTypeInfoString< T_UCoordinatesCuda >() << "\n"
    << "T_Color            = " << getTypeInfoString< T_Color            >() << "\n"
    << "T_Flags            = " << getTypeInfoString< T_Flags            >() << "\n"
    << "T_Id               = " << getTypeInfoString< T_Id               >() << "\n"
    << "T_Lattice          = " << getTypeInfoString< T_Lattice          >() << "\n";

    /* write out macro definition configuration */
    mLog( "Info" ) << "[" << __FILENAME__ << "::initialize] Macro configurations:\n"
    << " - working with bit-packed temporary lattice is : " << met.getPacking().getBitPackingOn() << "\n"
    << " - using " << met.getPacking().getNBufferedTmpLatticeOn() << " with "  << mnLatticeTmpBuffers << " temporary lattices to calculate on a fresh one while the rest is still cleaning in another stream\n"
    << " - using GPU for initializations: " << met.isONGPUForOverhead()<< "\n"
    
    #if defined( MAX_CONNECTIVITY )
        << " - maximum connectivity is " << MAX_CONNECTIVITY << "\n"
    #endif	
    ;
    if ( mUsePeriodicMonomerSorting )
        mLog( "Info" ) << " - periodically sorting the monomers inside the array in respect to their spatial position every " << mnStepsBetweenSortings << "-th step to increase cache hit rates\n";

    mLog( "Info" ) << "use randomg number generator Saru " << "\n";

    auto constexpr maxBoxSize = ( 1llu << ( CHAR_BIT * sizeof( T_CoordinateCuda ) ) );
    if ( mBoxX > maxBoxSize || mBoxY > maxBoxSize || mBoxZ > maxBoxSize )
    {
        std::stringstream msg;
        msg << "The box size is limited to " << maxBoxSize << " in each direction"
            << ", because of the chosen type for T_Coordinate = "
            << getTypeInfoString< T_Coordinate >() << ", but the chose box size is: ("
            << mBoxX << "," << mBoxY << "," << mBoxZ << ")!\n"
            << "Please change T_Coordinate to a larger type if you want to simulate this setup.";
        throw std::runtime_error( msg.str() );
    }

    /**
     * "When you execute asynchronous CUDA commands without specifying
     * a stream, * the runtime uses the default stream. Before CUDA 7,
     * the default stream is  * a special stream which implicitly
     * synchronizes with all other streams on the device."
     * @see https://devblogs.nvidia.com/gpu-pro-tip-cuda-7-streams-simplify-concurrency/
     */
    if ( mStream == 0 )
        CUDA_ERROR( cudaStreamCreate( &mStream ) );

    { decltype( dcBoxX      ) x = mBoxX     ; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxX     , &x, sizeof(x) ) ); }
    { decltype( dcBoxY      ) x = mBoxY     ; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxY     , &x, sizeof(x) ) ); }
    { decltype( dcBoxZ      ) x = mBoxZ     ; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxZ     , &x, sizeof(x) ) ); }
    { decltype( dcBoxXM1    ) x = mBoxXM1   ; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxXM1   , &x, sizeof(x) ) ); }
    { decltype( dcBoxYM1    ) x = mBoxYM1   ; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxYM1   , &x, sizeof(x) ) ); }
    { decltype( dcBoxZM1    ) x = mBoxZM1   ; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxZM1   , &x, sizeof(x) ) ); }
    { decltype( dcBoxXLog2  ) x = mBoxXLog2 ; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxXLog2 , &x, sizeof(x) ) ); }
    { decltype( dcBoxXYLog2 ) x = mBoxXYLog2; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxXYLog2, &x, sizeof(x) ) ); }


    if (met.isONGPUForOverhead()){
        mPolymerSystem->pushAsync();
        mNeighbors    ->pushAsync();
    }

    initializeBondTable();
    initializeSpeciesSorting(); /* using miNewToi and miToiNew the monomers are mapped to be sorted by species */
    checkMonomerReorderMapping();
    initializeSortedNeighbors();
    initializeSortedMonomerPositions();
//     checkSystem();
    initializeLattices();
    
//     if ( mAge != 0 )
//         doSpatialSorting();
    boxCheck.initialize( mIsPeriodicX, mIsPeriodicY, mIsPeriodicZ );
    
    /* Saru, IntHash and Philox don't need any particular initialization */

    CUDA_ERROR( cudaGetDevice( &miGpuToUse ) );
    CUDA_ERROR( cudaGetDeviceProperties( &mCudaProps, miGpuToUse ) );
}


template< typename T_UCoordinateCuda >
void UpdaterGPUScBFM< T_UCoordinateCuda >::copyBondSet
( int dx, int dy, int dz, bool bondForbidden )
{
  checkBondVector.addBondVector(dx,dy,dz,bondForbidden);
}

template< typename T_UCoordinateCuda >
void UpdaterGPUScBFM< T_UCoordinateCuda >::setNrOfAllMonomers( T_Id const rnAllMonomers )
{
    if ( mnAllMonomers != 0 )
    {
        std::stringstream msg;
        msg << "[" << __FILENAME__ << "::setNrOfAllMonomers] "
            << "Number of Monomers already set to " << mnAllMonomers << "!\n";
        throw std::runtime_error( msg.str() );
    }

    mnAllMonomers = rnAllMonomers;
    mAttributeSystem.resize( mnAllMonomers   );
    mNeighbors     = new MirroredVector< MonomerEdges  >( mnAllMonomers );
    mPolymerSystem = new MirroredVector< T_Coordinates >( mnAllMonomers );
    std::memset( mNeighbors->host, 0, mNeighbors->nBytes );
}

template< typename T_UCoordinateCuda >
void UpdaterGPUScBFM< T_UCoordinateCuda >::setPeriodicity
(
    bool const isPeriodicX,
    bool const isPeriodicY,
    bool const isPeriodicZ
)
{
    mIsPeriodicX = isPeriodicX;
    mIsPeriodicY = isPeriodicY;
    mIsPeriodicZ = isPeriodicZ;
}

template< typename T_UCoordinateCuda >
void UpdaterGPUScBFM< T_UCoordinateCuda >::setAttribute( T_Id i, int32_t attribute ){ mAttributeSystem.at(i) = attribute; }

/**
 * @todo add a runtime error for coordinates exceeding the maximum type range
 */
template< typename T_UCoordinateCuda >
void UpdaterGPUScBFM< T_UCoordinateCuda >::setMonomerCoordinates
(
    T_Id          const i,
    T_Coordinate  const x,
    T_Coordinate  const y,
    T_Coordinate  const z
)
{
    mPolymerSystem->host[i].x = x;
    mPolymerSystem->host[i].y = y;
    mPolymerSystem->host[i].z = z;
}
template< typename T_UCoordinateCuda > int32_t UpdaterGPUScBFM< T_UCoordinateCuda >::getMonomerPositionInX( T_Id i ){ return mPolymerSystem->host[i].x; }
template< typename T_UCoordinateCuda > int32_t UpdaterGPUScBFM< T_UCoordinateCuda >::getMonomerPositionInY( T_Id i ){ return mPolymerSystem->host[i].y; }
template< typename T_UCoordinateCuda > int32_t UpdaterGPUScBFM< T_UCoordinateCuda >::getMonomerPositionInZ( T_Id i ){ return mPolymerSystem->host[i].z; }

template< typename T_UCoordinateCuda >
void UpdaterGPUScBFM< T_UCoordinateCuda >::setConnectivity
(
    T_Id const iMonomer1,
    T_Id const iMonomer2
)
{
    /* @todo add check whether the bond already exists */
    /* Could also add the inversio, but the bonds are a non-directional graph */
    auto const iNew = mNeighbors->host[ iMonomer1 ].size++;
    if ( iNew > MAX_CONNECTIVITY-1 )
    {
        std::stringstream msg;
        msg << "[" << __FILENAME__ << "::setConnectivity" << "] "
            << "The maximum amount of bonds per monomer (" << MAX_CONNECTIVITY
            << ") has been exceeded!\n";
        throw std::invalid_argument( msg.str() );
    }
    mNeighbors->host[ iMonomer1 ].neighborIds[ iNew ] = iMonomer2;
}

template< typename T_UCoordinateCuda >
void UpdaterGPUScBFM< T_UCoordinateCuda >::setLatticeSize
(
    T_BoxSize const boxX,
    T_BoxSize const boxY,
    T_BoxSize const boxZ
)
{
    if ( mBoxX == boxX && mBoxY == boxY && mBoxZ == boxZ )
        return;

    if ( ! ( inRange< T_Coordinate >( boxX ) &&
             inRange< T_Coordinate >( boxY ) &&
             inRange< T_Coordinate >( boxZ )    ) )
    {
        std::stringstream msg;
        msg << "[" << __FILENAME__ << "::setLatticeSize" << "] "
            << "The box size (" << boxX << "," << boxY << "," << boxZ
            << ") is larger than the internal integer data type for "
            << "representing positions allow! (" << std::numeric_limits< T_Coordinate >::min()
            << " <= size <= " << std::numeric_limits< T_Coordinate >::max() << ")";
        throw std::invalid_argument( msg.str() );
    }

    mBoxX   = boxX;
    mBoxY   = boxY;
    mBoxZ   = boxZ;
    mBoxXM1 = boxX-1;
    mBoxYM1 = boxY-1;
    mBoxZM1 = boxZ-1;

    /* determine log2 for mBoxX and mBoxX * mBoxY to be used for bitshifting
     * the indice instead of multiplying ... WHY??? I don't think it is faster,
     * but much less readable */
    mBoxXLog2  = 0; auto dummy = mBoxX ; while ( dummy >>= 1 ) ++mBoxXLog2;
    mBoxXYLog2 = 0; dummy = mBoxX*mBoxY; while ( dummy >>= 1 ) ++mBoxXYLog2;
    if ( mBoxX != ( 1u << mBoxXLog2 ) || mBoxX * boxY != ( 1u << mBoxXYLog2 ) )
    {
        std::stringstream msg;
        msg << "[" << __FILENAME__ << "::setLatticeSize" << "] "
            << "Could not determine value for bit shift. "
            << "Check whether the box size is a power of 2! ( "
            << "boxX=" << mBoxX << " =? 2^" << mBoxXLog2 << " = " << ( 1 << mBoxXLog2 )
            << ", boxX*boY=" << mBoxX * mBoxY << " =? 2^" << mBoxXYLog2
            << " = " << ( 1 << mBoxXYLog2 ) << " )\n";
        throw std::runtime_error( msg.str() );
    }
}

/**
 * Uses mPolymerSystemSortedOld and mPolymerSystemSorted and finds overflows
 * assuming a given physical known maximum movement since the old data.
 * Both inputs are assumed to be on the gpu!
 * If so, then the overflow is reversed if it happened because of data type
 * overflow and/or is counted into mviPolymerSystemSortedVirtualBox if it
 * happened because of the periodic boundary condition of the box.
 */
template< typename T_UCoordinateCuda >
void UpdaterGPUScBFM< T_UCoordinateCuda >::findAndRemoveOverflows( bool copyToHost )
{
    /**
     * find jumps and "deapply" them. We just have to find jumps larger than
     * the number of time steps calculated assuming the monomers can only move
     * 1 cell per time step per direction (meaning this also works with
     * diagonal moves!)
     * If for example the box size is 32, but we also calculate with uint8,
     * then the particles may seemingly move not only bei +-32, but also by
     * +-256, but in both cases the particle actually only moves one virtual
     * box.
     * E.g. the particle was at 0 moved to -1 which was mapped to 255 because
     * uint8 overflowed, but the box size is 64, then deltaMove=255 and we
     * need to subtract 3*64. This only works if the box size is a multiple of
     * the type maximum number (256). I.e. in any sane environment if the box
     * size is a power of two, which was a requirement anyway already.
     * Actually, as the position is just calculated as +-1 without any wrapping,
     * the only way for jumps to happen is because of type overflows.
     */

    if (met.isONGPUForOverhead() ){
      auto const nThreads = 128;
      auto const nBlocks  = ceilDiv( mnMonomersPadded, nThreads );
      /* the padding values do not change, so we can simply let the threads
      * calculate them without worries and save the loop over the species */
      kernelTreatOverflows< T_UCoordinateCuda >
      <<< nBlocks, nThreads, 0, mStream >>>(
	  mPolymerSystemSortedOld         ->gpu,
	  mPolymerSystemSorted            ->gpu,
	  mviPolymerSystemSortedVirtualBox->gpu,
	  mnMonomersPadded
      );
      if ( copyToHost )
      {
	  mPolymerSystemSorted            ->pop();
	  mviPolymerSystemSortedVirtualBox->pop();
      }
    }else{
      mPolymerSystemSorted            ->popAsync();
      mviPolymerSystemSortedVirtualBox->popAsync();
      CUDA_ERROR( cudaStreamSynchronize( mStream ) );

      size_t nPrintInfo = 10;
      for ( T_Id i = 0u; i < mnAllMonomers; ++i )
      {
	  auto const r0tmp = mPolymerSystemSortedOld         ->host[ miToiNew->host[i] ];
	  auto const r1tmp = mPolymerSystemSorted            ->host[ miToiNew->host[i] ];
	  auto const ivtmp = mviPolymerSystemSortedVirtualBox->host[ miToiNew->host[i] ];
	  T_UCoordinateCuda r0[3] = { r0tmp.x, r0tmp.y, r0tmp.z };
	  T_UCoordinateCuda r1[3] = { r1tmp.x, r1tmp.y, r1tmp.z };
	  T_Coordinate      iv[3] = { ivtmp.x, ivtmp.y, ivtmp.z };

	  std::vector< T_BoxSize > const boxSizes = { mBoxX, mBoxY, mBoxZ };
	  auto constexpr boxSizeCudaType = 1ll << ( sizeof( T_UCoordinateCuda ) * CHAR_BIT );
	  for ( auto iCoord = 0u; iCoord < 3u; ++iCoord )
	  {
	      assert( boxSizeCudaType >= boxSizes[ iCoord ] );
	      //assert( nMonteCarloSteps <= boxSizeCudaType / 2 );
	      //assert( nMonteCarloSteps <= std::min( std::min( mBoxX, mBoxY ), mBoxZ ) / 2 );
	      auto const deltaMove = r1[ iCoord ] - r0[ iCoord ];
	      if ( std::abs( (int) deltaMove ) > boxSizeCudaType / 2 )
	      {
		  if ( nPrintInfo > 0 )
		  {
		      --nPrintInfo;
		      mLog( "Info" )
			  << i << " " << char( 'x' + iCoord ) << ": "
			  << (int) r0[ iCoord ] << " -> " << (int) r1[ iCoord ] << " -> "
			  << T_Coordinate( r1[ iCoord ] - ( boxSizeCudaType - boxSizes[ iCoord ] ) )
			  << "\n";
		  }
		  r1[ iCoord ] -= boxSizeCudaType - boxSizes[ iCoord ];
		  iv[ iCoord ] -= deltaMove > decltype(deltaMove)(0) ? 1 : -1;
	      }
	  }
	  mPolymerSystemSorted->host[ miToiNew->host[i] ].x = r1[0];
	  mPolymerSystemSorted->host[ miToiNew->host[i] ].y = r1[1];
	  mPolymerSystemSorted->host[ miToiNew->host[i] ].z = r1[2];
	  mviPolymerSystemSortedVirtualBox->host[ miToiNew->host[i] ].x = iv[0];
	  mviPolymerSystemSortedVirtualBox->host[ miToiNew->host[i] ].y = iv[1];
	  mviPolymerSystemSortedVirtualBox->host[ miToiNew->host[i] ].z = iv[2];
      }
      mviPolymerSystemSortedVirtualBox->pushAsync();
    }
}
template< typename T_UCoordinateCuda >
void UpdaterGPUScBFM< T_UCoordinateCuda >::checkLatticeOccupation() const
{
   /* note that std::vector< bool > already uses bitpacking!
     * We'd have to watch out when erasing that array with memset! */
    std::vector< uint8_t > lattice( mBoxX * mBoxY * mBoxZ, 0 );

    /**
     * Test for excluded volume by setting all lattice points and count the
     * toal lattice points occupied. If we have overlap this will be smaller
     * than calculated for zero overlap!
     * mPolymerSystem only stores the lower left front corner of the 2x2x2
     * monomer cube. Use that information to set all 8 cells in the lattice
     * to 'occupied'
     */
    /*
     Lattice is an array of size Box_X*Box_Y*Box_Z. PolymerSystem holds the monomer positions which I strongly guess are supposed to be in the range 0<=x<Box_X. If I see correctly, then this part checks for excluded volume by occupying a 2x2x2 cube for each monomer in Lattice and then counting the total occupied cells and compare it to the theoretical value of nMonomers * 8. But Lattice seems to be too small for that kinda usage! I.e. for two particles, one being at x=0 and the other being at x=Box_X-1 this test should return that the excluded volume condition is not met! Therefore the effective box size is actually (Box_X-1,Box_X-1,Box_Z-1) which in my opinion should be a bug ??? */
    for ( T_Id i = 0; i < mnAllMonomers; ++i )
    {
        int32_t const & x = mPolymerSystem->host[i].x;
        int32_t const & y = mPolymerSystem->host[i].y;
        int32_t const & z = mPolymerSystem->host[i].z;
        /**
         * @verbatim
         *           ...+---+---+
         *     ...'''   | 6 | 7 |
         *    +---+---+ +---+---+    y
         *    | 2 | 3 | | 4 | 5 |    ^ z
         *    +---+---+ +---+---+    |/
         *    | 0 | 1 |   ...'''     +--> x
         *    +---+---+'''
         * @endverbatim
         */
        lattice[ met.getCurve().linearizeBoxVectorIndex( x  , y  , z   ) ] = 1; /* 0 */
        lattice[ met.getCurve().linearizeBoxVectorIndex( x+1, y  , z   ) ] = 1; /* 1 */
        lattice[ met.getCurve().linearizeBoxVectorIndex( x  , y+1, z   ) ] = 1; /* 2 */
        lattice[ met.getCurve().linearizeBoxVectorIndex( x+1, y+1, z   ) ] = 1; /* 3 */
        lattice[ met.getCurve().linearizeBoxVectorIndex( x  , y  , z+1 ) ] = 1; /* 4 */
        lattice[ met.getCurve().linearizeBoxVectorIndex( x+1, y  , z+1 ) ] = 1; /* 5 */
        lattice[ met.getCurve().linearizeBoxVectorIndex( x  , y+1, z+1 ) ] = 1; /* 6 */
        lattice[ met.getCurve().linearizeBoxVectorIndex( x+1, y+1, z+1 ) ] = 1; /* 7 */
    }
    /* check total occupied cells inside lattice to ensure that the above
     * transfer went without problems. Note that the number will be smaller
     * if some monomers overlap!
     * Could also simply reduce mLattice with +, I think, because it only
     * cotains 0 or 1 ??? */
    unsigned nOccupied = 0;
    for ( uint32_t i = 0u; i < mBoxX * mBoxY * mBoxZ; ++i )
        nOccupied += lattice[i] != 0;
    if ( ! ( nOccupied == mnAllMonomers * 8 ) )
    {
        std::stringstream msg;
        msg << "[" << __FILENAME__ << "::~checkSystem" << "] "
            << "Occupation count in mLattice is wrong! Expected 8*nMonomers="
            << 8 * mnAllMonomers << " occupied cells, but got " << nOccupied;
        throw std::runtime_error( msg.str() );
    }
}
template< typename T_UCoordinateCuda >
void UpdaterGPUScBFM< T_UCoordinateCuda >::checkBonds() const
{
  
    /**
     * Check bonds i.e. that |dx|<=3 and whether it is allowed by the given
     * bond set
     */
    for ( T_Id i = 0; i < mnAllMonomers; ++i )
    for ( unsigned iNeighbor = 0; iNeighbor < mNeighbors->host[i].size; ++iNeighbor )
    {
        /* calculate the bond vector between the neighbor and this particle
         * neighbor - particle = ( dx, dy, dz ) */
        auto const neighbor = mPolymerSystem->host[ mNeighbors->host[i].neighborIds[ iNeighbor ] ];
        auto dx = (int) neighbor.x - (int) mPolymerSystem->host[i].x;
        auto dy = (int) neighbor.y - (int) mPolymerSystem->host[i].y;
        auto dz = (int) neighbor.z - (int) mPolymerSystem->host[i].z;
        /* with this uncommented, we can ignore if a monomer jumps over the
         * whole box range or T_UCoordinateCuda range */
        /*
        #ifndef NDEBUG
            auto constexpr nLongestBond = 8u;
            assert( mBoxX >= nLongestBond );
            assert( mBoxY >= nLongestBond );
            assert( mBoxZ >= nLongestBond );
        #endif
        dx %= mBoxX; if ( dx < -int( mBoxX )/ 2 ) dx += mBoxX; if ( dx > (int) mBoxX / 2 ) dx -= mBoxX;
        dy %= mBoxY; if ( dy < -int( mBoxY )/ 2 ) dy += mBoxY; if ( dy > (int) mBoxY / 2 ) dy -= mBoxY;
        dz %= mBoxZ; if ( dz < -int( mBoxZ )/ 2 ) dz += mBoxZ; if ( dz > (int) mBoxZ / 2 ) dz -= mBoxZ;
        */

        int erroneousAxis = -1;
        if ( ! ( -3 <= dx && dx <= 3 ) ) erroneousAxis = 0;
        if ( ! ( -3 <= dy && dy <= 3 ) ) erroneousAxis = 1;
        if ( ! ( -3 <= dz && dz <= 3 ) ) erroneousAxis = 2;
        if ( erroneousAxis >= 0 || checkBondVector( dx, dy, dz ) )
        {
            std::stringstream msg;
            msg << "[" << __FILENAME__ << "::checkSystem] ";
            if ( erroneousAxis > 0 )
                msg << "Invalid " << char( 'X' + erroneousAxis ) << "-Bond: ";
            if ( checkBondVector( dx, dy, dz ) )
                msg << "This particular bond is forbidden: ";
            msg << "(" << dx << "," << dy<< "," << dz << ") between monomer "
                << i << " at (" << mPolymerSystem->host[i].x << ","
                                << mPolymerSystem->host[i].y << ","
                                << mPolymerSystem->host[i].z << ") and monomer "
                << mNeighbors->host[i].neighborIds[ iNeighbor ] << " at ("
                << neighbor.x << "," << neighbor.y << "," << neighbor.z << ")"
                << std::endl;
             throw std::runtime_error( msg.str() );
        }
    }
  
}

/**
 * Checks for excluded volume condition and for correctness of all monomer bonds
 * Beware, it useses and thereby thrashes mLattice. Might be cleaner to declare
 * as const and malloc and free some temporary buffer, but the time ...
 * https://randomascii.wordpress.com/2014/12/10/hidden-costs-of-memory-allocation/
 * "In my tests, for sizes ranging from 8 MB to 32 MB, the cost for a new[]/delete[] pair averaged about 7.5 μs (microseconds), split into ~5.0 μs for the allocation and ~2.5 μs for the free."
 *  => ~40k cycles
 */
template< typename T_UCoordinateCuda >
void UpdaterGPUScBFM< T_UCoordinateCuda >::checkSystem() const
{
    if ( ! mLog.isActive( "Check" ) )
        return;
    checkLatticeOccupation();
    checkBonds();
}

template< typename T_UCoordinateCuda  >
void UpdaterGPUScBFM< T_UCoordinateCuda >::launch_CheckSpecies(
    const size_t nBlocks, const size_t nThreads, 
    const size_t iSpecies, const size_t iOffsetLatticeTmp, 
    const uint64_t seed)
{
  kernelSimulationScBFMCheckSpecies< T_UCoordinateCuda > 
  <<< nBlocks, nThreads, 0, mStream >>>(                
      mPolymerSystemSorted->gpu,                                     
      mPolymerFlags->gpu + mviSubGroupOffsets[ iSpecies ],           
      mviSubGroupOffsets[ iSpecies ],                                
      mLatticeTmp->gpu + iOffsetLatticeTmp,                          
      mNeighborsSorted->gpu + mNeighborsSortedInfo.getMatrixOffsetElements( iSpecies ), 
      mNeighborsSortedInfo.getMatrixPitchElements( iSpecies ),       
      mNeighborsSortedSizes->gpu + mviSubGroupOffsets[ iSpecies ],   
      mnElementsInGroup[ iSpecies ],                                 
      seed, 
      mGlobalIterator,                                         
      mLatticeOut->texture,
      boxCheck, 
      met,
      checkBondVector
  );
  mGlobalIterator++;
}

template< typename T_UCoordinateCuda  >
void UpdaterGPUScBFM< T_UCoordinateCuda >::launch_CheckReactiveSpecies(
    const size_t nBlocks, const size_t nThreads, 
    const size_t iSpecies, const size_t iOffsetLatticeTmp, 
    const uint64_t seed, uint32_t AASpeciesFlag,
    cudaTextureObject_t const texAllowedToMoveInSpecies )
{
  kernelSimulationScBFMCheckReactiveSpecies< T_UCoordinateCuda > 
  <<< nBlocks, nThreads, 0, mStream >>>(                
      mPolymerSystemSorted->gpu,                                     
      mPolymerFlags->gpu + mviSubGroupOffsets[ iSpecies ],           
      mviSubGroupOffsets[ iSpecies ],                                
      mLatticeTmp->gpu + iOffsetLatticeTmp,                          
      mNeighborsSorted->gpu + mNeighborsSortedInfo.getMatrixOffsetElements( iSpecies ), 
      mNeighborsSortedInfo.getMatrixPitchElements( iSpecies ),       
      mNeighborsSortedSizes->gpu + mviSubGroupOffsets[ iSpecies ],   
      mnElementsInGroup[ iSpecies ],                                 
      seed, 
      mGlobalIterator,                                         
      mLatticeOut->texture,
      boxCheck, 
      met,
      checkBondVector,
      texAllowedToMoveInSpecies,
      AASpeciesFlag
  );
  mGlobalIterator++;
}

template< typename T_UCoordinateCuda  >
void UpdaterGPUScBFM< T_UCoordinateCuda >::launch_countFilteredPerform(
    const size_t nBlocks, const size_t nThreads, 
    const size_t iSpecies, cudaTextureObject_t texLatticeTmp, 
    unsigned long long int * dpFiltered )
{
    kernelCountFilteredPerform< T_UCoordinateCuda >
    <<< nBlocks, nThreads, 0, mStream >>>(
        mPolymerSystemSorted->gpu + mviSubGroupOffsets[ iSpecies ],
        mPolymerFlags->gpu + mviSubGroupOffsets[ iSpecies ],
        mLatticeOut->gpu,
        mnElementsInGroup[ iSpecies ],
        texLatticeTmp,
        dpFiltered,
        met
    );
}

template< typename T_UCoordinateCuda  >
void UpdaterGPUScBFM< T_UCoordinateCuda >::launch_CountFilteredCheck(
    const size_t nBlocks, const size_t nThreads, 
    const size_t iSpecies, cudaTextureObject_t texLatticeTmp, 
    unsigned long long int * dpFiltered, const size_t iOffsetLatticeTmp )
{
    kernelCountFilteredCheck< T_UCoordinateCuda >
    <<< nBlocks, nThreads, 0, mStream >>>(                 
    mPolymerSystemSorted->gpu,                         
    mPolymerFlags->gpu + mviSubGroupOffsets[ iSpecies ],
    mviSubGroupOffsets[ iSpecies ],                     
    mLatticeTmp->gpu + iOffsetLatticeTmp,               
    mNeighborsSorted->gpu + mNeighborsSortedInfo.getMatrixOffsetElements( iSpecies ), 
    mNeighborsSortedInfo.getMatrixPitchElements( iSpecies ),   
    mNeighborsSortedSizes->gpu + mviSubGroupOffsets[ iSpecies ], 
    mnElementsInGroup[ iSpecies ],                             
    mLatticeOut->texture,                                     
    dpFiltered,
    boxCheck,
    met,
    checkBondVector
    );
}

template< typename T_UCoordinateCuda  >
void UpdaterGPUScBFM< T_UCoordinateCuda >::launch_PerformSpecies(
    const size_t nBlocks, const size_t nThreads, 
    const size_t iSpecies, cudaTextureObject_t texLatticeTmp )
{
    kernelSimulationScBFMPerformSpecies< T_UCoordinateCuda >
    <<< nBlocks, nThreads, 0, mStream >>>(
        mPolymerSystemSorted->gpu + mviSubGroupOffsets[ iSpecies ],
        mPolymerFlags->gpu + mviSubGroupOffsets[ iSpecies ],
        mLatticeOut->gpu,
        mnElementsInGroup[ iSpecies ],
        texLatticeTmp, met 
    );
}

template< typename T_UCoordinateCuda  >
void UpdaterGPUScBFM< T_UCoordinateCuda >::launch_PerformSpeciesAndApply(
    const size_t nBlocks, const size_t nThreads, 
    const size_t iSpecies, cudaTextureObject_t texLatticeTmp )
{
    kernelSimulationScBFMPerformSpeciesAndApply< T_UCoordinateCuda >
    <<< nBlocks, nThreads, 0, mStream >>>(
       mPolymerSystemSorted->gpu + mviSubGroupOffsets[ iSpecies ],
        mPolymerFlags->gpu + mviSubGroupOffsets[ iSpecies ],
        mLatticeOut->gpu,
        mnElementsInGroup[ iSpecies ],
        texLatticeTmp, met 
    );
}

template< typename T_UCoordinateCuda  >
void UpdaterGPUScBFM< T_UCoordinateCuda >::launch_ZeroArraySpecies(
    const size_t nBlocks, const size_t nThreads, 
    const size_t iSpecies )
{
    kernelSimulationScBFMZeroArraySpecies< T_UCoordinateCuda >
    <<< nBlocks, nThreads, 0, mStream >>>(
        mPolymerSystemSorted->gpu + mviSubGroupOffsets[ iSpecies ],
        mPolymerFlags->gpu + mviSubGroupOffsets[ iSpecies ],
        mLatticeTmp->gpu,
        mnElementsInGroup[ iSpecies ], met 
    );
}
#include <LeMonADEGPU/utility/AutomaticThreadChooser.h>
template< typename T_UCoordinateCuda  >
void UpdaterGPUScBFM< T_UCoordinateCuda >::runSimulationOnGPU
(
    uint32_t const nMonteCarloSteps
)
{
    std::clock_t const t0 = std::clock();

    CUDA_ERROR( cudaStreamSynchronize( mStream ) ); // finish e.g. initializations
    CUDA_ERROR( cudaMemcpy( mPolymerSystemSortedOld->gpu, mPolymerSystemSorted->gpu, mPolymerSystemSortedOld->nBytes, cudaMemcpyDeviceToDevice ) );
    auto const nSpecies = mnElementsInGroup.size();
// 
    /**
     * Statistics (min, max, mean, stddev) on filtering. Filtered because of:
     *   0: bonds, 1: volume exclusion, 2: volume exclusion (parallel)
     * These statistics are done for each group separately
     */
    std::vector< std::vector< double > > sums, sums2, mins, maxs, ns;
    std::vector< unsigned long long int > vFiltered;
    unsigned long long int * dpFiltered = NULL;
    auto constexpr nFilters = 5;
    if ( mLog.isActive( "Stats" ) )
    {
        sums .resize( nSpecies, std::vector< double >( nFilters, 0             ) );
        sums2.resize( nSpecies, std::vector< double >( nFilters, 0             ) );
        mins .resize( nSpecies, std::vector< double >( nFilters, mnAllMonomers ) );
        maxs .resize( nSpecies, std::vector< double >( nFilters, 0             ) );
        ns   .resize( nSpecies, std::vector< double >( nFilters, 0             ) );
        /* ns needed because we need to know how often each group was advanced */
        vFiltered.resize( nFilters );
        CUDA_ERROR( cudaMalloc( &dpFiltered, nFilters * sizeof( *dpFiltered ) ) );
        CUDA_ERROR( cudaMemsetAsync( (void*) dpFiltered, 0, nFilters * sizeof( *dpFiltered ), mStream ) );
    }
     /**
     * should never lead to a speedup and non power of twos, e.g. 196 even,
     * won't be able to perfectly fill out the shared multi processor.
     * Also, automatically determine whether cudaMemset is faster or not (after
     * we found the best threads per block configuration
     * note: test example best configuration was 128 threads per block and use
     *       the cudaMemset version instead of the third kernel
     */
    std::vector< int > vnThreadsToTry;
    for ( auto nThreads = mCudaProps.warpSize; nThreads <= mCudaProps.maxThreadsPerBlock; nThreads *= 2 )
        vnThreadsToTry.push_back( nThreads );
    assert( vnThreadsToTry.size() > 0 );
    struct SpeciesBenchmarkData
    {
        /* 2 vectors of double for measurements with and without cudaMemset */
        std::vector< std::vector< float > > timings;
        std::vector< std::vector< int   > > nRepeatTimings;
        int iBestThreadCount;
        bool useCudaMemset;
        bool decidedAboutThreadCount;
        bool decidedAboutCudaMemset;
    };
    std::vector< SpeciesBenchmarkData > benchmarkInfo( nSpecies, SpeciesBenchmarkData{
        std::vector< std::vector< float > >( 2 /* true or false */,
            std::vector< float >( vnThreadsToTry.size(),
                std::numeric_limits< float >::infinity() ) ),
        std::vector< std::vector< int   > >( 2 /* true or false */,
            std::vector< int   >( vnThreadsToTry.size(),
            2 /* repeat 2 time, i.e. execute three times */ ) ),
        2, true, true, true
    } );
    cudaEvent_t tOneGpuLoop0, tOneGpuLoop1;
    cudaEventCreate( &tOneGpuLoop0 );
    cudaEventCreate( &tOneGpuLoop1 );

    std::vector< uint64_t > nSpeciesChosen( nSpecies ,0 );

    /* run simulation */
    for ( uint32_t iStep = 0; iStep < nMonteCarloSteps; ++iStep, ++mAge )
    {
        if ( mUsePeriodicMonomerSorting && ( mAge % mnStepsBetweenSortings == 0 ) )
        {
            mLog( "Info" ) << "Resorting at age / step " << mAge << "\n";
            doSpatialSorting();
        }
        if ( useOverflowChecks )
        {
            /**
             * for uint8_t we have to check for overflows every 127 steps, as
             * for 128 steps we couldn't say whether it actually moved 128 steps
             * or whether it moved 128 steps in the other direction and was wrapped
             * to be equal to the hypothetical monomer above
             */
            auto constexpr boxSizeCudaType = 1ll << ( sizeof( T_UCoordinateCuda ) * CHAR_BIT );
            auto constexpr nStepsBetweenOverflowChecks = boxSizeCudaType / 2 - 1;
            if ( iStep != 0 && iStep % nStepsBetweenOverflowChecks == 0 )
            {
                findAndRemoveOverflows( false );
                CUDA_ERROR( cudaMemcpyAsync( mPolymerSystemSortedOld->gpu,
                    mPolymerSystemSorted->gpu, mPolymerSystemSortedOld->nBytes,
                    cudaMemcpyDeviceToDevice, mStream ) );
            }
        }
        /* one Monte-Carlo step:
         *  - tries to move on average all particles one time
         *  - each particle could be touched, not just one group */
        for ( uint32_t iSubStep = 0; iSubStep < nSpecies; ++iSubStep )
        {

	  
	    auto const iStepTotal = iStep * nSpecies + iSubStep;
	    auto  iOffsetLatticeTmp = ( iStepTotal % mnLatticeTmpBuffers )
		* ( mBoxX * mBoxY * mBoxZ * sizeof( mLatticeTmp->gpu[0] ));
	    if (met.getPacking().getBitPackingOn()) 
	      iOffsetLatticeTmp /= CHAR_BIT;
	    auto texLatticeTmp = mvtLatticeTmp[ iStepTotal % mnLatticeTmpBuffers ];

	    if (met.getPacking().getNBufferedTmpLatticeOn()) {
                iOffsetLatticeTmp = 0u;
                texLatticeTmp = mLatticeTmp->texture;
	    }


            /* randomly choose which monomer group to advance */
            auto const iSpecies = randomNumbers.r250_rand32() % nSpecies;
            auto const seed     = randomNumbers.r250_rand32();
            auto const nThreads = vnThreadsToTry.at( benchmarkInfo[ iSpecies ].iBestThreadCount );
            auto const nBlocks  = ceilDiv( mnElementsInGroup[ iSpecies ], nThreads );
            auto const needToBenchmark = ! (
                benchmarkInfo[ iSpecies ].decidedAboutThreadCount &&
                benchmarkInfo[ iSpecies ].decidedAboutCudaMemset );
            auto const useCudaMemset = benchmarkInfo[ iSpecies ].useCudaMemset;
            if ( needToBenchmark )
                cudaEventRecord( tOneGpuLoop0, mStream );

            nSpeciesChosen[ iSpecies ] += 1;

            /*
            if ( iStep < 3 )
                mLog( "Info" ) << "Calling Check-Kernel for species " << iSpecies << " for uint32_t * " << (void*) mNeighborsSorted->gpu << " + " << mNeighborsSortedInfo.getMatrixOffsetElements( iSpecies ) << " = " << (void*)( mNeighborsSorted->gpu + mNeighborsSortedInfo.getMatrixOffsetElements( iSpecies ) ) << " with pitch " << mNeighborsSortedInfo.getMatrixPitchElements( iSpecies ) << "\n";
            */

	    launch_CheckSpecies(nBlocks, nThreads, iSpecies, iOffsetLatticeTmp, seed);

	    /** The counting kernel can come after the Check-kernel, because the
             * Check-kernel only modifies the polymer flags which it does not
             * read itself. It actually wouldn't work else, because the count
             * kernel needs to query the drawn direction 
	     * @todo find the bug !!!
	     */
	    //somehow it does not work with the boxCheck method. 
// 	    launch_CountFilteredCheck(nBlocks,nThreads,iSpecies, texLatticeTmp, dpFiltered, iOffsetLatticeTmp);

	    if ( mLog.isActive( "Stats" ) )
		launch_countFilteredPerform(nBlocks,nThreads, iSpecies, texLatticeTmp, dpFiltered);

	    if ( useCudaMemset )
		launch_PerformSpeciesAndApply(nBlocks, nThreads, iSpecies, texLatticeTmp);
	    else
		launch_PerformSpecies(nBlocks,nThreads,iSpecies,texLatticeTmp);

	    if ( useCudaMemset ){
		if(met.getPacking().getNBufferedTmpLatticeOn()){
		    /* we only need to delete when buffers will wrap around and
			* on the last loop, so that on next runSimulationOnGPU
			* call mLatticeTmp is clean */
		    if ( ( iStepTotal % mnLatticeTmpBuffers == 0 ) ||
			( iStep == nMonteCarloSteps-1 && iSubStep == nSpecies-1 ) )
		    {
			cudaMemsetAsync( (void*) mLatticeTmp->gpu, 0, mLatticeTmp->nBytes, mStream );
		    }
		}else
		    mLatticeTmp->memsetAsync(0);
	    }
	    else
		launch_ZeroArraySpecies(nBlocks,nThreads,iSpecies);

	    if ( needToBenchmark )
	    {
		auto & info = benchmarkInfo[ iSpecies ];
		cudaEventRecord( tOneGpuLoop1, mStream );
		cudaEventSynchronize( tOneGpuLoop1 );
		float milliseconds = 0;
		cudaEventElapsedTime( & milliseconds, tOneGpuLoop0, tOneGpuLoop1 );
		auto const iThreadCount = info.iBestThreadCount;
		auto & oldTiming = info.timings.at( useCudaMemset ).at( iThreadCount );
		oldTiming = std::min( oldTiming, milliseconds );

		mLog( "Info" )
		<< "Using " << nThreads << " threads (" << nBlocks << " blocks)"
		<< " and using " << ( useCudaMemset ? "cudaMemset" : "kernelZeroArray" )
		<< " for species " << iSpecies << " took " << milliseconds << "ms\n";

		auto & nStillToRepeat = info.nRepeatTimings.at( useCudaMemset ).at( iThreadCount );
		if ( nStillToRepeat > 0 )
		    --nStillToRepeat;
		else if ( ! info.decidedAboutThreadCount )
		{
		    /* if not the first timing, then decide whether we got slower,
			* i.e. whether we found the minimum in the last step and
			* have to roll back */
		    if ( iThreadCount > 0 )
		    {
			if ( info.timings.at( useCudaMemset ).at( iThreadCount-1 ) < milliseconds )
			{
			    --info.iBestThreadCount;
			    info.decidedAboutThreadCount = true;
			}
			else
			    ++info.iBestThreadCount;
		    }
		    else
			++info.iBestThreadCount;
		    /* if we can't increment anymore, then we are finished */
		    assert( (size_t) info.iBestThreadCount <= vnThreadsToTry.size() );
		    if ( (size_t) info.iBestThreadCount == vnThreadsToTry.size() )
		    {
			--info.iBestThreadCount;
			info.decidedAboutThreadCount = true;
		    }
		    if ( info.decidedAboutThreadCount )
		    {
			/* then in the next term try out changing cudaMemset
			    * version to custom kernel version (or vice-versa) */
			if ( ! info.decidedAboutCudaMemset )
			    info.useCudaMemset = ! info.useCudaMemset;
			mLog( "Info" )
			<< "Using " << vnThreadsToTry.at( info.iBestThreadCount )
			<< " threads per block is the fastest for species "
			<< iSpecies << ".\n";
		    }
		}
		else if ( ! info.decidedAboutCudaMemset )
		{
		    info.decidedAboutCudaMemset = true;
		    if ( info.timings.at( ! useCudaMemset ).at( iThreadCount ) < milliseconds )
			info.useCudaMemset = ! useCudaMemset;
		    if ( info.decidedAboutCudaMemset )
		    {
			mLog( "Info" )
			<< "Using " << ( info.useCudaMemset ? "cudaMemset" : "kernelZeroArray" )
			<< " is the fastest for species " << iSpecies << ".\n";
		    }
		}
	    }

            if ( mLog.isActive( "Stats" ) )
            {
                CUDA_ERROR( cudaMemcpyAsync( (void*) &vFiltered.at(0), (void*) dpFiltered,
                    nFilters * sizeof( *dpFiltered ), cudaMemcpyDeviceToHost, mStream ) );
                CUDA_ERROR( cudaStreamSynchronize( mStream ) );
                CUDA_ERROR( cudaMemsetAsync( (void*) dpFiltered, 0, nFilters * sizeof( *dpFiltered ), mStream ) );

                for ( auto iFilter = 0u; iFilter < nFilters; ++iFilter )
                {
                    double const x = vFiltered.at( iFilter );
                    sums .at( iSpecies ).at( iFilter ) += x;
                    sums2.at( iSpecies ).at( iFilter ) += x*x;
                    mins .at( iSpecies ).at( iFilter ) = std::min( mins.at( iSpecies ).at( iFilter ), x );
                    maxs .at( iSpecies ).at( iFilter ) = std::max( maxs.at( iSpecies ).at( iFilter ), x );
                    ns   .at( iSpecies ).at( iFilter ) += 1;
                }
            }
        } // iSubstep
    } // iStep
    if ( mLog.isActive( "Stats" ) && dpFiltered != NULL )
    {
        if ( mnElementsInGroup.size() <= 8 )
        for ( auto iSpecies = 0u; iSpecies < nSpecies; ++iSpecies )
            mLog( "Stats" ) << "Group " << char( 'A' + iSpecies ) << " was chosen " << nSpeciesChosen[ iSpecies ] << " times\n";

        mLog( "Stats" ) << "Filter analysis. Format:\n" << "Filter Reason: min | mean +- stddev | max\n";
        std::map< int, std::string > filterNames;
        filterNames[0] = "Box Boundaries";
        filterNames[1] = "Invalid Bonds";
        filterNames[2] = "Volume Exclusion";
        filterNames[3] = "! Invalid Bonds && Volume Exclusion";
        filterNames[4] = "! Invalid Bonds && ! Volume Exclusion && Parallel Volume Exclusion";

        if ( mnElementsInGroup.size() <= 8 )
        for ( auto iGroup = 0u; iGroup < mnElementsInGroup.size(); ++iGroup )
        {
            mLog( "Stats" ) << "\n=== Group " << char( 'A' + iGroup ) << " (" << mnElementsInGroup[ iGroup ] << ") ===\n";
            for ( auto iFilter = 0u; iFilter < nFilters; ++iFilter )
            {
                double const nRepeats = ns   .at( iGroup ).at( iFilter );
                double const mean     = sums .at( iGroup ).at( iFilter ) / nRepeats;
                double const sum2     = sums2.at( iGroup ).at( iFilter ) / nRepeats;
                double const min      = mins .at( iGroup ).at( iFilter );
                double const max      = maxs .at( iGroup ).at( iFilter );
                double const stddev   = std::sqrt( nRepeats/(nRepeats-1.) * ( sum2 - mean * mean ) );
                mLog( "Stats" )
                    << filterNames[iFilter] << ": "
                    << min  << "(" << 100. * min  / mnElementsInGroup[ iGroup ] << "%) | "
                    << mean << "(" << 100. * mean / mnElementsInGroup[ iGroup ] << "%) +- "
                    << stddev << " | "
                    << max  << "(" << 100. * max  / mnElementsInGroup[ iGroup ] << "%)\n";
            }
            if ( sums.at( iGroup ).at(0) != 0 )
                mLog( "Stats" ) << "The value for remeaining particles after first kernel will be wrong if we have non-periodic boundary conditions (todo)!\n";
            auto const nAvgFilteredKernel1 = (double)( sums.at( iGroup ).at(1) + sums.at( iGroup ).at(3) ) / ns.at( iGroup ).at(3);
            mLog( "Stats" ) << "Remaining after 1st kernel: " << mnElementsInGroup[ iGroup ] - nAvgFilteredKernel1 << "(" << 100. * ( mnElementsInGroup[ iGroup ] - nAvgFilteredKernel1 ) / mnElementsInGroup[ iGroup ] << "%)\n";
            auto const nAvgFilteredKernel2 = (double) sums.at( iGroup ).at(4) / ns.at( iGroup ).at(4);
            auto const percentageMoved = ( mnElementsInGroup[ iGroup ] - nAvgFilteredKernel1 - nAvgFilteredKernel2 ) / mnElementsInGroup[ iGroup ];
            mLog( "Stats" ) << "For parallel collisions it's interesting to give the percentage of sorted particles in relation to whose who can actually still move, not in relation to ALL particles\n"
                << "    Third kernel gets " << mnElementsInGroup[ iGroup ] << " monomers, but first kernel (bonds, box, volume exclusion) already filtered " << nAvgFilteredKernel1 << "(" << 100. * nAvgFilteredKernel1 / mnElementsInGroup[ iGroup ] << "%) which the 2nd kernel has to refilter again (if no stream compaction is used).\n"
                << "    Then from the remaining " << mnElementsInGroup[ iGroup ] - nAvgFilteredKernel1 << "(" << 100. * ( mnElementsInGroup[ iGroup ] - nAvgFilteredKernel1 ) / mnElementsInGroup[ iGroup ] << "%) the 2nd kernel filters out another " << nAvgFilteredKernel2 << " particles which in relation to the particles which actually still could move before is: " << 100. * nAvgFilteredKernel2 / ( mnElementsInGroup[ iGroup ] - nAvgFilteredKernel1 ) << "% and in relation to the total particles: " << 100. * nAvgFilteredKernel2 / mnElementsInGroup[ iGroup ] << "%\n"
                << "    Therefore in total (all three kernels) and on average (multiple salves of three kernels) " << ( mnElementsInGroup[ iGroup ] - nAvgFilteredKernel1 - nAvgFilteredKernel2 ) << "(" << 100. * percentageMoved << "%) particles can be moved per step. I.e. it takes on average " << 1. / percentageMoved << " Monte-Carlo steps per monomer until a monomer actually changes position.\n";
        }

        /* Give total statistics not separated by species. Currently this only works if species all have the same number of monomers because the statistics are done over the actual filtered cound instead of the percentages */
        if ( mnElementsInGroup.size() > 1 )
        {
            mLog( "Stats" ) << "\n\n==== Total Filtering Statistics over all Species ====\n";
            std::vector< double > totSums ( nFilters, 0 );
            std::vector< double > totSums2( nFilters, 0 );
            std::vector< double > totMins ( nFilters, 0 );
            std::vector< double > totMaxs ( nFilters, 0 );
            std::vector< double > totNs   ( nFilters, 0 );
            for ( auto iFilter = 0u; iFilter < nFilters; ++iFilter )
            {
                for ( auto iGroup = 0u; iGroup < mnElementsInGroup.size(); ++iGroup )
                {
                    totSums [ iFilter ] += sums [ iGroup ][ iFilter ];
                    totSums2[ iFilter ] += sums2[ iGroup ][ iFilter ];
                    totNs   [ iFilter ] += ns   [ iGroup ][ iFilter ];
                    totMins [ iFilter ] += mins [ iGroup ][ iFilter ];
                    totMaxs [ iFilter ] += maxs [ iGroup ][ iFilter ];
                }
                double const nRepeats = nMonteCarloSteps; // totNs   [ iFilter ];
                double const mean     = totSums [ iFilter ] / nRepeats;
                double const sum2     = totSums2[ iFilter ] / nRepeats;
                double const min      = totMins [ iFilter ];
                double const max      = totMaxs [ iFilter ];
                double const stddev   = std::sqrt( nRepeats/(nRepeats-1.) * ( sum2 - mean * mean ) );
                // sum2 - mean*mean can get negative ... meh
                /*
                std::cout << "sums2 = " << sums2[0][iFilter] << "," << sums2[1][iFilter] << "\n";
                std::cout << "mean  = " << mean << ", nRepeats = " << nRepeats << "\n";
                std::cout << "sum2 - mean * mean  = " << sum2 - mean * mean << "\n";
                std::cout << "sum2 - mean * mean  = " << sum2 - mean * mean << "\n";
                */
                mLog( "Stats" )
                    << filterNames[iFilter] << ": "
                    << min  << "(" << 100. * min  / mnAllMonomers << "%) | "
                    << mean << "(" << 100. * mean / mnAllMonomers << "%) +- "
                    << stddev << " | "
                    << max  << "(" << 100. * max  / mnAllMonomers << "%)\n";
            }
        }

        CUDA_ERROR( cudaFree( dpFiltered ) );
    }
    doCopyBack();
    checkSystem(); // no-op if "Check"-level deactivated
    std::clock_t const t1 = std::clock();
    double const dt = float(t1-t0) / CLOCKS_PER_SEC;
    mLog( "Info" )
    << "run time (GPU): " << nMonteCarloSteps << "\n"
    << "mcs = " << nMonteCarloSteps  << "  speed [performed monomer try and move/s] = MCS*N/t: "
    << nMonteCarloSteps * ( mnAllMonomers / dt )  << "     runtime[s]:" << dt << "\n";
}

template< typename T_UCoordinateCuda >
void UpdaterGPUScBFM< T_UCoordinateCuda >::doCopyBack()
{
    mLog( "Info" ) << "UpdaterGPUScBFM< T_UCoordinateCuda >::doCopyBackConnectivity() \n";
    doCopyBackMonomerPositions();
//     mLog( "Info" ) << "UpdaterGPUScBFM< T_UCoordinateCuda >::doCopyBackConnectivity() \n";
//     doCopyBackConnectivity();
}
template< typename T_UCoordinateCuda >
void UpdaterGPUScBFM< T_UCoordinateCuda >::doCopyBackMonomerPositions()
{
    /* all MCS are done- copy information back from GPU to host */
    if ( mLog.isActive( "Check" ) )
    {
        mLatticeTmp->pop( false ); // sync
        size_t nOccupied = 0;
        for ( size_t i = 0u; i < mLatticeTmp->nElements; ++i )
            nOccupied += mLatticeTmp->host[i] != 0;
        if ( nOccupied != 0 )
        {
            std::stringstream msg;
            msg << "latticeTmp occupation (" << nOccupied << ") should be 0! Exiting ...\n";
            throw std::runtime_error( msg.str() );
        }
    }
    mLog( "Info" ) << "Start copying back  \n";
    if ( ! met.isONGPUForOverhead()){
      if ( mUsePeriodicMonomerSorting )
      {
	  miNewToi->popAsync();
	  miToiNew->popAsync(); /* needed for findAndRemoveOverflows, but only if met.isONGPUForOverhead() not set */
	  CUDA_ERROR( cudaStreamSynchronize( mStream ) );
	  checkMonomerReorderMapping();
      }
    }
    if ( useOverflowChecks )
    {
        findAndRemoveOverflows( false );
    }
    if(met.isONGPUForOverhead() ){
      auto const nThreads = 128;
      auto const nBlocksP = ceilDiv( mnMonomersPadded, nThreads );
      kernelUndoPolymerSystemSorting< T_UCoordinateCuda >
      <<< nBlocksP, nThreads, 0, mStream >>>
      (
	  mPolymerSystemSorted            ->gpu,
	  mviPolymerSystemSortedVirtualBox->gpu,
	  miNewToi                        ->gpu,
	  mPolymerSystem                  ->gpu,
	  mnMonomersPadded
      );
      mPolymerSystem->pop();
    }else{
      mPolymerSystemSorted->pop();
      mviPolymerSystemSortedVirtualBox->pop();
      /* untangle reordered array so that LeMonADE can use it again */
      for ( T_Id i = 0u; i < mnAllMonomers; ++i )
      {
	  auto const pTarget = mPolymerSystemSorted->host + miToiNew->host[i];
	  if ( i < 10 )
	      mLog( "Info" ) << "Copying back " << i << " from " << miToiNew->host[i] << "\n";
	  mPolymerSystem->host[i].x = pTarget->x + mviPolymerSystemSortedVirtualBox->host[ miToiNew->host[i] ].x * mBoxX;
	  mPolymerSystem->host[i].y = pTarget->y + mviPolymerSystemSortedVirtualBox->host[ miToiNew->host[i] ].y * mBoxY;
	  mPolymerSystem->host[i].z = pTarget->z + mviPolymerSystemSortedVirtualBox->host[ miToiNew->host[i] ].z * mBoxZ;
	  mPolymerSystem->host[i].w = pTarget->w;
      }
    }
}
template< typename T_UCoordinateCuda >
uint32_t UpdaterGPUScBFM< T_UCoordinateCuda >::getNumLinks(uint32_t MonID)
{
  return  mNeighbors->host[ MonID ].size;
}

template< typename T_UCoordinateCuda >
uint32_t UpdaterGPUScBFM< T_UCoordinateCuda >::getNeighborIdx(uint32_t MonID, uint32_t BondID)
{
  return  mNeighbors->host[ MonID ].neighborIds[BondID]; 
}

// __global__ void 

template< typename T_UCoordinateCuda >
void UpdaterGPUScBFM< T_UCoordinateCuda >::doCopyBackConnectivity()
{
    mNeighborsSortedSizes->pop();
    miNewToi->pop();
//     miToiNew->pop();
    mNeighborsSorted->pop();
    //update the bonds between monomers:
    mLog ( "Info" ) << "Copied  mNeighborsSortedSizes, miNewToi and mNeighborsSorted. Start writing connectivity:\n";
    size_t iSpecies = 0u;
    /* iterate over sorted instead of unsorted array so that calculating
      * the current species we are working on is easier */
    for ( size_t i = 0u; i < miNewToi->nElements; ++i )
    {
	
	/* check if we are already working on a new species */
	if ( iSpecies+1 < mviSubGroupOffsets.size() &&
	      i >= mviSubGroupOffsets[ iSpecies+1 ] )
	{
	  mLog( "Check" ) <<"Increase number of species by one at " << i <<"\n";
	    ++iSpecies;
	}
	auto iOld(miNewToi->host[i]);
	/* skip over padded indices */
	if ( iOld >= mnAllMonomers )
	    continue;
	/* actually to the sorting / copying and conversion */
	auto const pitch = mNeighborsSortedInfo.getMatrixPitchElements( iSpecies );
	mNeighbors->host[iOld ].size = (uint32_t)mNeighborsSortedSizes->host[i];

	for ( size_t j = 0; j < (uint32_t)mNeighborsSortedSizes->host[i]; j++ )
	{
	    mNeighbors->host[ iOld ].neighborIds[j] = miNewToi->host        [ 
	                                              mNeighborsSorted->host[ 
	                                                mNeighborsSortedInfo.getMatrixOffsetElements( iSpecies ) + j * pitch + ( i - mviSubGroupOffsets[ iSpecies ] ) 
									    ] ];
// 	    if (iSpecies < 2 && mLog("Info").isActive() )
// 	    {
// 	      mLog("Info")  << "New= " << i << " iOld= " <<iOld << " nLink= " << (uint32_t)mNeighborsSortedSizes->host[i] 
// 			    << " GPUNeighbor= " << mNeighborsSorted->host[ mNeighborsSortedInfo.getMatrixOffsetElements( iSpecies ) + j * pitch + ( i - mviSubGroupOffsets[ iSpecies ] ) ]
// 			    << " CPUNeighbor= " << mNeighbors->host[ iOld ].neighborIds[j] << "\n" ;
// 	    }
	}
    }
}
/**
 * GPUScBFM::initialize and run and cleanup should be usable on
 * repeat. Which means we need to destruct everything created in
 * GPUScBFM::initialize, which encompasses setLatticeSize,
 * setNrOfAllMonomers and initialize. Currently this includes all allocs,
 * so we can simply call destruct.
 */
template< typename T_UCoordinateCuda >
void UpdaterGPUScBFM< T_UCoordinateCuda >::cleanup()
{
    this->destruct();
    cudaDeviceSynchronize();
    cudaProfilerStop();
}

template class UpdaterGPUScBFM< uint8_t  >;
template class UpdaterGPUScBFM< uint16_t >;
template class UpdaterGPUScBFM< uint32_t >;
template class UpdaterGPUScBFM<  int16_t >;
template class UpdaterGPUScBFM<  int32_t >;
