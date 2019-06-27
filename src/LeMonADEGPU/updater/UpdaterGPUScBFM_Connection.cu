

/*
 * UpdaterGPUScBFM_Connection.cu
 *
 *  Created on: 27.06.2019
 *      Authors: Toni Mueller
 */

#include <LeMonADEGPU/updater/UpdaterGPUScBFM_Connection.h>
#include <LeMonADEGPU/updater/UpdaterGPUScBFM_AB_Type.h>
#include <LeMonADEGPU/utility/cudacommon.hpp>
#include <LeMonADEGPU/core/Method.h>

using T_Flags            = UpdaterGPUScBFM_AB_Type< uint8_t >::T_Flags      ;
using T_Lattice          = UpdaterGPUScBFM_AB_Type< uint8_t >::T_Lattice    ;
  
/**
 * We want to introduce connections during the simulation. 
 */
template< typename T_UCoordinateCuda >
__global__ void kernelCheckSpeciesConnection
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
    Method              const              met
){
   uint32_t rn;
    int iGrid = 0;
    for ( auto iMonomer = blockIdx.x * blockDim.x + threadIdx.x;
          iMonomer < nMonomers; iMonomer += gridDim.x * blockDim.x, ++iGrid )
    {
        auto const r0 = dpPolymerSystem[ iOffset + iMonomer ];
        if ( iGrid % 1 == 0 ) //for what is this  
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
//	if (  ! checkBonds<T_UCoordinateCuda>(dpNeighborsSizes, iMonomer, dpNeighbors, rNeighborsPitchElements, dpPolymerSystem, r1 ) )
	{
	    direction += T_Flags(8);
	    met.getPacking().bitPackedSet(dpLatticeTmp, met.getCurve().linearizeBoxVectorIndex( r1.x, r1.y, r1.z ));
	}
        dpPolymerFlags[ iMonomer ] = direction;
    }
}

template< typename T_UCoordinateCuda > 
UpdaterGPUScBFM_Connection<T_UCoordinateCuda>::UpdaterGPUScBFM_Connection():
BaseClass(){
    /**
     * Log control.
     * Note that "Check" controls not the output, but the actualy checks
     * If a checks needs to always be done, then do that check and declare
     * the output as "Info" log level
     */
    mLog.file( __FILENAME__ );
    mLog.deactivate( "Check"     );
    mLog.deactivate( "Error"     );
    mLog.deactivate( "Info"      );
    mLog.deactivate( "Stats"     );
    mLog.deactivate( "Warning"   );
};


template <typename T_UCoordinateCuda>
void UpdaterGPUScBFM_Connection<T_UCoordinateCuda>::initialize()
{
  BaseClass::initialize();
   
}

template class UpdaterGPUScBFM_Connection< uint8_t  >;
template class UpdaterGPUScBFM_Connection< uint16_t >;
template class UpdaterGPUScBFM_Connection< uint32_t >;
template class UpdaterGPUScBFM_Connection<  int16_t >;
template class UpdaterGPUScBFM_Connection<  int32_t >;


