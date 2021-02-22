

/*
 * UpdaterGPUScBFM_AB_Breaking.cu
 *
 *  Created on: 27.06.2019
 *      Authors: Toni Mueller
 */

#include <LeMonADEGPU/updater/UpdaterGPUScBFM_AB_Breaking.h>
// #include <LeMonADEGPU/updater/UpdaterGPUScBFM.h>
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

using T_Id               = UpdaterGPUScBFM< uint8_t >::T_Id         ;
__device__ __constant__ double dcBreakingProbability     ;  // functionality of cross links 
__global__ void kernel_BreakConnections
(
    uint8_t           * const              dpNeighborsSizes               ,
    T_Id              * const              dpNeighbors                    ,
    uint32_t            const              rNeighborsMatrixOffsetMonomer  ,
    uint32_t            const              rNeighborsMatrixOffsetPartner  ,
    uint32_t            const              rNeighborsPitchElementsMonomer ,
    uint32_t            const              rNeighborsPitchElementsPartner ,
    uint32_t 		const 		   iOffsetCrossLinks              ,
    uint32_t 		const 		   iOffsetChains                  ,
    T_Id                const              nMonomers                      ,
    uint64_t            const              rSeed                          ,
    uint64_t            const              rGlobalIteration               ,
    T_Id              * const              dBreaksID1                     ,
    T_Id              * const              dBreaksID2
)
{
    double rn;
    int iGrid;
    for ( uint32_t iMonomer = blockIdx.x * blockDim.x + threadIdx.x;
          iMonomer < nMonomers; iMonomer += gridDim.x * blockDim.x, iGrid++)
    {   
      if (iGrid %1 ==0 ){
	Saru rng(rGlobalIteration,iMonomer,rSeed);
        auto nNeighbors(dpNeighborsSizes[iOffsetCrossLinks + iMonomer]);
	auto iBond(0);
        for ( uint32_t i=0; i < nNeighbors; i++ ) 
	{
	  rn =rng.rng_d(); 
	  if (rn < dcBreakingProbability)//break it! 
	  {
	    auto const iGlobalNeighbor = dpNeighbors[ rNeighborsMatrixOffsetMonomer + iBond * rNeighborsPitchElementsMonomer + iMonomer ];
	    auto neighborBond(dpNeighborsSizes[iGlobalNeighbor]);

	    for (uint32_t j=neighborBond; j >0 ; j-- ) //start from the top because in the most cases the second bond is the bond we need... 
	    {
	      auto tmpNeighbor(dpNeighbors[ rNeighborsMatrixOffsetPartner + (j-1) * rNeighborsPitchElementsPartner + (iGlobalNeighbor-iOffsetChains) ]);
// 	      printf("iGlobalNeighbor= %d, iOffset=%d , j=%d tmpNeighbor=%d MonID=%d rn=%f exp(-E)=%f  \n",iGlobalNeighbor,iOffsetChains, (j-1),  tmpNeighbor,  iMonomer+iOffsetCrossLinks, rn, dcBreakingProbability );
	      if ( tmpNeighbor == iMonomer+iOffsetCrossLinks ) 
	      {
		neighborBond=(j-1);
		break;
	      }
	    }
	    dpNeighborsSizes[ iGlobalNeighbor ]--;
	    dpNeighbors[ rNeighborsMatrixOffsetPartner + neighborBond * rNeighborsPitchElementsPartner + (iGlobalNeighbor-iOffsetChains) ]=dpNeighbors[ rNeighborsMatrixOffsetPartner + dpNeighborsSizes[iGlobalNeighbor ] * rNeighborsPitchElementsPartner + (iGlobalNeighbor-iOffsetChains) ];
	    dpNeighborsSizes[iOffsetCrossLinks + iMonomer ]--;
	    dpNeighbors[ rNeighborsMatrixOffsetMonomer + iBond * rNeighborsPitchElementsMonomer + iMonomer ]=dpNeighbors[ rNeighborsMatrixOffsetMonomer + dpNeighborsSizes[iOffsetCrossLinks + iMonomer ] * rNeighborsPitchElementsMonomer + iMonomer ];
	    dBreaksID1[i] = iMonomer+1-iOffsetCrossLinks;
	    dBreaksID2[i] = iGlobalNeighbor+1-iOffsetChains;
	  }else 
	    iBond++;
	}
      }
    }
}

template< typename T_UCoordinateCuda > 
void UpdaterGPUScBFM_AB_Breaking<T_UCoordinateCuda>::launch_BreakConnections(
	  const size_t nBlocks, const size_t nThreads, 
	  const size_t iSpeciesCrossLink, const size_t iSpeciesChain, 
	  const uint64_t seed)
{
  kernel_BreakConnections<<<nBlocks,nThreads,0,mStream>>>
  (
      mNeighborsSortedSizes->gpu , 
      mNeighborsSorted->gpu,
      mNeighborsSortedInfo.getMatrixOffsetElements( iSpeciesCrossLink ),
      mNeighborsSortedInfo.getMatrixOffsetElements( iSpeciesChain ),
      mNeighborsSortedInfo.getMatrixPitchElements( iSpeciesCrossLink ),
      mNeighborsSortedInfo.getMatrixPitchElements( iSpeciesChain ),
      mviSubGroupOffsets[ iSpeciesCrossLink ],   
      mviSubGroupOffsets[ iSpeciesChain ],
      mnElementsInGroup[ iSpeciesCrossLink ],                       
      seed, 
      hGlobalIterator,
      dBreaksID1->gpu,
      dBreaksID2->gpu
  );
//   CUDA_ERROR(cudaDeviceSynchronize());
  hGlobalIterator++;
  tracker.trackBreaks( dBreaksID1->gpu, dBreaksID2->gpu, nReactiveMonomersCrossLinks+1, 
    miNewToi->gpu,mviSubGroupOffsets[ iSpeciesCrossLink ], mviSubGroupOffsets[ iSpeciesChain ], mAge,
    mPolymerSystemSorted, mviPolymerSystemSortedVirtualBox);
//   tracker.trackBreaks( dBreaks->gpu, nReactiveMonomersCrossLinks+1, miNewToi->gpu, mAge);
 
}
template< typename T_UCoordinateCuda > 
UpdaterGPUScBFM_AB_Breaking<T_UCoordinateCuda>::UpdaterGPUScBFM_AB_Breaking():
BaseClass()  ,
dBreaksID1(NULL),
dBreaksID2(NULL)
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
void UpdaterGPUScBFM_AB_Breaking<T_UCoordinateCuda>::destruct(){
      
    DeleteMirroredObject deletePointer;
    deletePointer( dBreaksID1       , "dBreaksID1"        );
    deletePointer( dBreaksID2       , "dBreaksID2"        );
    if ( deletePointer.nBytesFreed > 0 )
    {
        mLog( "Info" )
            << "Freed a total of "
            << prettyPrintBytes( deletePointer.nBytesFreed )
            << " on GPU and host RAM.\n";
    }
}
template< typename T_UCoordinateCuda > 
UpdaterGPUScBFM_AB_Breaking<T_UCoordinateCuda>::~UpdaterGPUScBFM_AB_Breaking()
{
  this->destruct();    
  destruct(); 
}

template< typename T_UCoordinateCuda >
void UpdaterGPUScBFM_AB_Breaking<T_UCoordinateCuda>::cleanup()
{
    this->destruct();    
    destruct();   
    cudaDeviceSynchronize();
    cudaProfilerStop();
    
}

template < typename T_UCoordinateCuda >
void UpdaterGPUScBFM_AB_Breaking<T_UCoordinateCuda>::initialize()
{
  BaseClass::initialize();
  
  double mBreakingProbability(exp(-energy));
  dBreaksID1 = new MirroredVector<T_Id>(nReactiveMonomersCrossLinks+1);
  dBreaksID2 = new MirroredVector<T_Id>(nReactiveMonomersCrossLinks+1);
  CUDA_ERROR( cudaMemcpyToSymbol( dcBreakingProbability, &mBreakingProbability, sizeof( mBreakingProbability ) ) );
  mLog("Info") << "Bond energy is " << energy << " which corresponds to a breaking probabilit of " << mBreakingProbability <<"\n" ;

}

template< typename T_UCoordinateCuda  >
void UpdaterGPUScBFM_AB_Breaking< T_UCoordinateCuda >::setBondEnergy(double energy_){energy=energy_;}


template< typename T_UCoordinateCuda  >
void UpdaterGPUScBFM_AB_Breaking< T_UCoordinateCuda >::runSimulationOnGPU
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
//       	tracker.setAge(mAge);
        if ( mUsePeriodicMonomerSorting && ( mAge % mnStepsBetweenSortings == 0 ) )
        {
            mLog( "Stats" ) << "Resorting at age / step " << mAge << "\n";
	    //this is not compatible with the rest of the code. Unfortunaltly I dont see why....
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

	    if (!diagMovesOn)  
	    {
		this-> template launch_CheckSpecies<6>(nBlocks, nThreads, iSpecies, iOffsetLatticeTmp, seed);
		if ( useCudaMemset )
		    launch_PerformSpeciesAndApply(nBlocks, nThreads, iSpecies, texLatticeTmp,seed );
		else
		    launch_PerformSpecies(nBlocks,nThreads,iSpecies,texLatticeTmp,seed );
	    }else 
	    {
		this-> template launch_CheckSpecies<18>(nBlocks, nThreads, iSpecies, iOffsetLatticeTmp, seed);
		if ( useCudaMemset )
		    launch_PerformSpeciesAndApply(nBlocks, nThreads, iSpecies, texLatticeTmp,seed );
		else
		    launch_PerformSpecies(nBlocks,nThreads,iSpecies,texLatticeTmp,seed );
	    }

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
            chooseThreads.analyze(iSpecies,mStream);
        } // iSubstep
	//connection process 
	//here we could again benchmark for a better performance gain...
        auto const nThreads = chooseThreads.getBestThread(ChainEndSpecies);
	auto const nBlocks  = ceilDiv( mnElementsInGroup[ ChainEndSpecies ], nThreads );
	launch_initializeReactiveLattice( nBlocks, nThreads, ChainEndSpecies);
	
	if (mLog( "Check" ).isActive())
	  checkReactiveLatticeOccupation();
	auto const nThreads_c = 128;
	auto const nBlocks_c  = ceilDiv( nReactiveMonomersCrossLinks, nThreads_c );
	auto const seed     = randomNumbers.r250_rand32();
        launch_CheckConnection(nBlocks_c,nThreads_c,CrossLinkSpecies, ChainEndSpecies,seed);
	launch_ApplyConnection(nBlocks_c,nThreads_c,CrossLinkSpecies, ChainEndSpecies);
	launch_resetReactiveLattice( nBlocks, nThreads, ChainEndSpecies);
	//breaking process
	launch_BreakConnections(nBlocks_c,nThreads_c,CrossLinkSpecies, ChainEndSpecies,seed);
	
    } // iStep
    
    std::clock_t const t1 = std::clock();
    double const dt = float(t1-t0) / CLOCKS_PER_SEC;
    mLog( "Info" )
    << "run time (GPU): " << nMonteCarloSteps << "\n"
    << "mcs = " << nMonteCarloSteps  << "  speed [performed monomer try and move/s] = MCS*N/t: "
    << nMonteCarloSteps * ( mnAllMonomers / dt )  << "     runtime[s]:" << dt << "\n";
    BaseClass::checkSystem(); // no-op if "Check"-level deactivated
    BaseClass::doCopyBack();
    tracker.dumpReactions();
}


template class UpdaterGPUScBFM_AB_Breaking< uint8_t  >;
template class UpdaterGPUScBFM_AB_Breaking< uint16_t >;
template class UpdaterGPUScBFM_AB_Breaking< uint32_t >;
template class UpdaterGPUScBFM_AB_Breaking<  int16_t >;
template class UpdaterGPUScBFM_AB_Breaking<  int32_t >;


