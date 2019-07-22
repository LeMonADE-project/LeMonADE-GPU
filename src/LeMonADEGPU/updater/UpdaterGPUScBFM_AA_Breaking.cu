

/*
 * UpdaterGPUScBFM_AA_Breaking.cu
 *
 *  Created on: 27.06.2019
 *      Authors: Toni Mueller
 */

#include <LeMonADEGPU/updater/UpdaterGPUScBFM_AA_Breaking.h>
// #include <LeMonADEGPU/updater/UpdaterGPUScBFM_AB_Type.h>
#include <LeMonADEGPU/utility/cudacommon.hpp>
#include <LeMonADEGPU/core/Method.h>
#include <LeMonADEGPU/utility/DeleteMirroredObject.h>
#include <cuda_profiler_api.h>              // cudaProfilerStop
#include <LeMonADEGPU/utility/AutomaticThreadChooser.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>

#include <extern/Fundamental/BitsCompileTime.hpp>

#include <LeMonADEGPU/utility/SelectiveLogger.hpp>
#include <LeMonADEGPU/utility/graphColoring.tpp>
#include <LeMonADEGPU/core/rngs/Saru.h>
#include <LeMonADEGPU/core/MonomerEdges.h>
#include <LeMonADEGPU/core/constants.cuh>
#include <LeMonADEGPU/feature/BoxCheck.h>
#include <LeMonADEGPU/core/Method.h>

#include <LeMonADEGPU/utility/DeleteMirroredObject.h>
#include <LeMonADEGPU/core/BondVectorSet.h>
#include <LeMonADEGPU/core/kernelConnection.h>
#include <LeMonADEGPU/utility/GPUConnectionTracker.h>
#include <math.h>

__device__ __constant__ double dcBreakingProbability     ;  // functionality of cross links 
__global__ void kernel_BreakConnections
(
    uint8_t           * const              dpNeighborsSizes               ,
    T_Id              * const              dpNeighbors                    ,
    uint32_t            const              rNeighborsPitchElements        ,
    uint32_t 		const 		   iOffset                        ,
    T_Id                const              nMonomers                      ,
    uint64_t            const              rSeed                          ,
    uint64_t            const              rGlobalIteration               ,
    uint8_t           * const              texAllowedToMove	          ,
    T_Id              * const              dBreaks
)
{
    double rn;
    int iGrid;
    for ( uint32_t iMonomer = blockIdx.x * blockDim.x + threadIdx.x;
          iMonomer < nMonomers; iMonomer += gridDim.x * blockDim.x, iGrid++)
    {   
      if ( texAllowedToMove[iMonomer] == 1 )
      {
	if (iGrid %1 ==0 ){
	  Saru rng(rGlobalIteration,iMonomer,rSeed);
	  rn =rng.rng_d(); 
	  if (rn < dcBreakingProbability)//break it! 
	  {
	    auto iBond1(0),iBond2(0),iGlobalNeighbor(0);
	    auto foundValues(false);
	    auto nNeighbors1(dpNeighborsSizes[iOffset + iMonomer]);
	    for (iBond1=0; iBond1< nNeighbors1;iBond1++)
	    {
	      iGlobalNeighbor = dpNeighbors[  iBond1 * rNeighborsPitchElements + iMonomer ];
	      for (iBond2=0; iBond2< dpNeighborsSizes[iGlobalNeighbor]; iBond2++)
	      {
		if (iOffset + iMonomer ==   dpNeighbors[  iBond2 * rNeighborsPitchElements + iGlobalNeighbor-iOffset ])
		{
		  foundValues=true;
		  break;
		}
	      }
	    }
	    if (foundValues)
	    {
	      dpNeighborsSizes[ iGlobalNeighbor ]--;
	      dpNeighbors[  iBond2 * rNeighborsPitchElements + (iGlobalNeighbor-iOffset) ]=dpNeighbors[  dpNeighborsSizes[iGlobalNeighbor ] * rNeighborsPitchElements + (iGlobalNeighbor-iOffset) ];
	      
	      dpNeighborsSizes[iOffset + iMonomer ]--;
	      dpNeighbors[ iBond1 * rNeighborsPitchElements + iMonomer ]=dpNeighbors[ dpNeighborsSizes[iOffset + iMonomer ] * rNeighborsPitchElements + iMonomer ];
	      
	      dBreaks[iMonomer+1] = iGlobalNeighbor+1;
	      
	      texAllowedToMove[iMonomer]=0;
	      texAllowedToMove[iGlobalNeighbor-iOffset]=0;
	    }
	  }
	}
      }
    }
}

template< typename T_UCoordinateCuda > 
void UpdaterGPUScBFM_AA_Breaking<T_UCoordinateCuda>::launch_BreakConnections(
	  const size_t nBlocks, const size_t nThreads, 
	  const size_t iSpecies, const uint64_t seed)
{
  kernel_BreakConnections<<<nBlocks,nThreads,0,mStream>>>
  (
      mNeighborsSortedSizes->gpu, 
      mNeighborsSorted->gpu + mNeighborsSortedInfo.getMatrixOffsetElements( iSpecies ),
      mNeighborsSortedInfo.getMatrixPitchElements( iSpecies ),
      mviSubGroupOffsets[ iSpecies ],
      mnElementsInGroup[ iSpecies ],                       
      seed, 
      mGlobalIterator,
      AAMonomerFlag->gpu,
      dBreaks->gpu
  );
//   CUDA_ERROR(cudaDeviceSynchronize());
  mGlobalIterator++;
  tracker.trackBreaks( dBreaks->gpu, nReactiveMonomers+1, miNewToi->gpu, mAge);
 
}
template< typename T_UCoordinateCuda > 
UpdaterGPUScBFM_AA_Breaking<T_UCoordinateCuda>::UpdaterGPUScBFM_AA_Breaking():
BaseClass()  ,
dBreaks(NULL)
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
    mLog.deactivate( "Info"      );
    mLog.deactivate( "Stats"     );
    mLog.deactivate( "Warning"   );
};
template< typename T_UCoordinateCuda > 
void UpdaterGPUScBFM_AA_Breaking<T_UCoordinateCuda>::destruct(){
      
    DeleteMirroredObject deletePointer;
    deletePointer( dBreaks       , "dBreaks"        );
    if ( deletePointer.nBytesFreed > 0 )
    {
        mLog( "Info" )
            << "Freed a total of "
            << prettyPrintBytes( deletePointer.nBytesFreed )
            << " on GPU and host RAM.\n";
    }
}
template< typename T_UCoordinateCuda > 
UpdaterGPUScBFM_AA_Breaking<T_UCoordinateCuda>::~UpdaterGPUScBFM_AA_Breaking()
{
  this->destruct();    
}

template< typename T_UCoordinateCuda >
void UpdaterGPUScBFM_AA_Breaking<T_UCoordinateCuda>::cleanup()
{
    BaseClass::destruct();
    this->destruct();    
    cudaDeviceSynchronize();
    cudaProfilerStop();
    
}

template < typename T_UCoordinateCuda >
void UpdaterGPUScBFM_AA_Breaking<T_UCoordinateCuda>::initialize()
{
  BaseClass::initialize();
  
  double mBreakingProbability(exp(-energy));
  std::cout <<"nReactiveMonomers+1 = "<< nReactiveMonomers+1 << "\n";
  dBreaks = new MirroredVector<T_Id>(nReactiveMonomers+1,mStream);
  CUDA_ERROR( cudaMemcpyToSymbol( dcBreakingProbability, &mBreakingProbability, sizeof( mBreakingProbability ) ) );
  mLog("Info") << "Bond energy is " << energy << " which corresponds to a breaking probabilit of " << mBreakingProbability <<"\n" ;

}

template< typename T_UCoordinateCuda  >
void UpdaterGPUScBFM_AA_Breaking< T_UCoordinateCuda >::setBondEnergy(double energy_){energy=energy_;}


template< typename T_UCoordinateCuda  >
void UpdaterGPUScBFM_AA_Breaking< T_UCoordinateCuda >::runSimulationOnGPU
(
    uint32_t const nMonteCarloSteps
)
{
  std::clock_t const t0 = std::clock();
    CUDA_ERROR( cudaStreamSynchronize( mStream ) ); // finish e.g. initializations
    CUDA_ERROR( cudaMemcpy( mPolymerSystemSortedOld->gpu, mPolymerSystemSorted->gpu, mPolymerSystemSortedOld->nBytes, cudaMemcpyDeviceToDevice ) );
    auto const nSpecies = mnElementsInGroup.size();
    AutomaticThreadChooser chooseThreads(nSpecies);
    chooseThreads.initialize(mCudaProps);
    std::vector< uint64_t > nSpeciesChosen( nSpecies ,0 );

    /* run simulation */
    for ( uint32_t iStep = 0; iStep < nMonteCarloSteps; ++iStep, ++mAge )
    {
        if ( mUsePeriodicMonomerSorting && ( mAge % mnStepsBetweenSortings == 0 ) )
        {
            mLog( "Stats" ) << "Resorting at age / step " << mAge << "\n";
//             doSpatialSorting();
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
            auto const nThreads = chooseThreads.getBestThread(iSpecies);
            auto const nBlocks  = ceilDiv( mnElementsInGroup[ iSpecies ], nThreads );
            auto const useCudaMemset = chooseThreads.useCudaMemset(iSpecies);
            chooseThreads.addRecord(iSpecies, mStream);

            nSpeciesChosen[ iSpecies ] += 1;
            
	    if (iSpecies != ChainEndSpecies )
	    {

	      launch_CheckSpecies(nBlocks, nThreads, iSpecies, iOffsetLatticeTmp, seed);

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
            }
            else 
            {
	      for(uint32_t n=0; n < 1; n++)
	      {

	      
		launch_CheckReactiveSpecies(nBlocks, nThreads, iSpecies, iOffsetLatticeTmp, seed, n, AAMonomerFlag->texture );
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
	      }
	      chooseThreads.analyze(iSpecies,mStream);
            }
        } // iSubstep
       
	//here we could again benchmark for a better performance gain...
        auto const nThreads = chooseThreads.getBestThread(ChainEndSpecies);
	auto const nBlocks  = ceilDiv( mnElementsInGroup[ ChainEndSpecies ], nThreads );
	launch_initializeReactiveLattice( nBlocks, nThreads, ChainEndSpecies);
	if (mLog( "Check" ).isActive())
	  checkReactiveLatticeOccupation();
	auto const nThreads_c = 128;
	auto const nBlocks_c  = ceilDiv( nReactiveMonomers, nThreads_c );
	auto const seed     = randomNumbers.r250_rand32();
        launch_CheckConnection(nBlocks_c,nThreads_c, ChainEndSpecies,seed);
	launch_ApplyConnection(nBlocks_c,nThreads_c, ChainEndSpecies);
	launch_resetReactiveLattice( nBlocks, nThreads, ChainEndSpecies);
	//breaks connections
	launch_BreakConnections(nBlocks_c,nThreads_c, ChainEndSpecies, seed);	
    } // iStep
    
    std::clock_t const t1 = std::clock();
    double const dt = float(t1-t0) / CLOCKS_PER_SEC;
    mLog( "Info" )
    << "run time (GPU): " << nMonteCarloSteps << "\n"
    << "mcs = " << nMonteCarloSteps  << "  speed [performed monomer try and move/s] = MCS*N/t: "
    << nMonteCarloSteps * ( mnAllMonomers / dt )  << "     runtime[s]:" << dt << "\n";
    BaseClass::doCopyBack();
    BaseClass::checkSystem(); // no-op if "Check"-level deactivated
    tracker.dumpReactions();    


}


template class UpdaterGPUScBFM_AA_Breaking< uint8_t  >;
template class UpdaterGPUScBFM_AA_Breaking< uint16_t >;
template class UpdaterGPUScBFM_AA_Breaking< uint32_t >;
template class UpdaterGPUScBFM_AA_Breaking<  int16_t >;
template class UpdaterGPUScBFM_AA_Breaking<  int32_t >;


