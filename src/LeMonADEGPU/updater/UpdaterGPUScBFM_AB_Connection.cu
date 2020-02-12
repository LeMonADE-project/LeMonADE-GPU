

/*
 * UpdaterGPUScBFM_AB_Connection.cu
 *
 *  Created on: 27.06.2019
 *      Authors: Toni Mueller
 */

#include <LeMonADEGPU/updater/UpdaterGPUScBFM_AB_Connection.h>
// #include <LeMonADEGPU/updater/UpdaterGPUScBFM.h>
#include <LeMonADEGPU/utility/cudacommon.hpp>
#include <LeMonADEGPU/core/Method.h>
#include <LeMonADEGPU/utility/DeleteMirroredObject.h>
#include <cuda_profiler_api.h>              // cudaProfilerStop
#include <LeMonADEGPU/utility/AutomaticThreadChooser.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>

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
#include <LeMonADEGPU/core/kernelConnection.h>
#include <LeMonADEGPU/utility/GPUConnectionTracker.h>

using T_Flags            = UpdaterGPUScBFM_AB_Connection< uint8_t >::T_Flags      ;
__device__ __constant__ uint32_t dcCrossLinkMaxNumLinks     ;  // functionality of cross links 
__device__ __constant__ uint32_t dcChainMaxNumLinks =  2    ;  // functionality of chain ends 
/**
 * @brief convinience function to print the box dimensions for the device constants 
 */
__global__ void CheckBoxDimensions()
{
printf("KernelCheckBoxDimensions: %d %d %d %d %d %d  %d %d \n",dcBoxX, dcBoxY, dcBoxZ,dcBoxXM1, dcBoxYM1,dcBoxZM1, dcBoxXLog2, dcBoxXYLog2 );
}

/**
 * @brief writes the ID of the chain ends on the lattice
 * @details The ID start at 1 and are shifted by and offset which is given
 * 	    by the previous species of monomers. 
 */
template< typename T_UCoordinateCuda >
__global__ void kernelUpdateReactiveLattice
(
    typename CudaVec4< T_UCoordinateCuda >::value_type
                        const * const __restrict__ dpPolymerSystem  ,
    uint32_t            const                      iOffset          ,
    T_Id                      * const __restrict__ dpReactiveLattice,
    T_Id                        const              nMonomers        ,
    Method                      const              met 
)
{
    for ( T_Id iMonomer = blockIdx.x * blockDim.x + threadIdx.x;
          iMonomer < nMonomers; iMonomer += gridDim.x * blockDim.x )
    {
        auto const r0 = dpPolymerSystem[ iOffset + iMonomer ];
	dpReactiveLattice[ met.getCurve().linearizeBoxVectorIndex( r0.x, r0.y, r0.z ) ] = ( iMonomer+1 );
    }
}
 /**
  * @brief convinience function to update the lattice occupation. 
  * @details We introduce such functions because then they can be used latter on from inheritate classes..
  */
template< typename T_UCoordinateCuda >
void UpdaterGPUScBFM_AB_Connection< T_UCoordinateCuda >::launch_initializeReactiveLattice(
  const size_t nBlocks , const size_t nThreads, const T_Id iSpecies )
{
  mLog ( "Check" ) <<"Start filling lattice with ones:  \n" ;
//   mLatticeIds->memset(0);
  mLog ( "Check" ) << "launch_initializeReactiveLattice:: iSpecies = " << iSpecies <<"\n"
		  << "launch_initializeReactiveLattice:: mviSubGroupOffsets[ iSpecies ] = "<< mviSubGroupOffsets[ iSpecies ]<<"\n"
		  << "launch_initializeReactiveLattice:: mnElementsInGroup[ iSpecies ] = "<< mnElementsInGroup[ iSpecies ]<<"\n";
  if ( false ){ //fill in cpu 
    mPolymerSystemSorted->pop();
    for (T_Id i =0; i < mnElementsInGroup[ iSpecies ]; i++)
    {
      auto const iMonomer(i+mviSubGroupOffsets[ iSpecies ]);
      auto const r(mPolymerSystemSorted->host[iMonomer]); 
      auto const Vector(met.getCurve().linearizeBoxVectorIndex(r.x,r.y,r.z));
      mLatticeIds->host[Vector]= iMonomer+1;
    }
    mLatticeIds->push(0);
    cudaStreamSynchronize( mStream );
  }else {
      kernelUpdateReactiveLattice<T_UCoordinateCuda><<<nBlocks,nThreads,0,mStream>>>(
	  mPolymerSystemSorted->gpu     ,            
	  mviSubGroupOffsets[ iSpecies ], 
	  mLatticeIds->gpu              ,
	  mnElementsInGroup[ iSpecies ] ,                        
	  met
      );
  }
}
/**
 * @brief writes 0 on the lattice where the chain ends are located 
 * @details Using this brings performance, because the lattice is dilute
 */
template< typename T_UCoordinateCuda >
__global__ void kernelResetReactiveLattice
(
    typename CudaVec4< T_UCoordinateCuda >::value_type
                        const * const __restrict__ dpPolymerSystem  ,
    uint32_t            const                      iOffset          ,
    T_Id                      * const __restrict__ dpReactiveLattice,
    T_Id                        const              nMonomers        ,
    Method                      const              met 
)
{
    for ( T_Id iMonomer = blockIdx.x * blockDim.x + threadIdx.x;
          iMonomer < nMonomers; iMonomer += gridDim.x * blockDim.x )
    {
        auto const r0 = dpPolymerSystem[ iOffset + iMonomer ];
	dpReactiveLattice[ met.getCurve().linearizeBoxVectorIndex( r0.x, r0.y, r0.z ) ] = 0;
    }
}
template< typename T_UCoordinateCuda >
void UpdaterGPUScBFM_AB_Connection< T_UCoordinateCuda >::launch_resetReactiveLattice(
  const size_t nBlocks , const size_t nThreads, const T_Id iSpecies )
{

//   mLatticeIds->memset(0,0); 

  if ( false ){ //fill in cpu 
    mPolymerSystemSorted->pop();
    for (T_Id i =0; i < mnElementsInGroup[ iSpecies ]; i++)
    {
      auto const iMonomer(i+mviSubGroupOffsets[ iSpecies ]);
      auto const r(mPolymerSystemSorted->host[iMonomer]); 
      auto const Vector(met.getCurve().linearizeBoxVectorIndex(r.x,r.y,r.z));
      mLatticeIds->host[Vector]= 0;
    }
    mLatticeIds->push(0);
    cudaStreamSynchronize( mStream );
  }else {
      kernelResetReactiveLattice<T_UCoordinateCuda><<<nBlocks,nThreads,0,mStream>>>(
	  mPolymerSystemSorted->gpu     ,            
	  mviSubGroupOffsets[ iSpecies ], 
	  mLatticeIds->gpu              ,
	  mnElementsInGroup[ iSpecies ] ,                        
	  met
      );
  }
}

/**
 * @brief Counts the number of occupied lattice sites.
 */
template< typename T_UCoordinateCuda  >
void UpdaterGPUScBFM_AB_Connection< T_UCoordinateCuda >::checkReactiveLatticeOccupation()  
{
  mLatticeIds->pop(0);
  uint32_t countLatticeEntries(0);
  for(T_Id x=0; x< mBoxX; x++ )
    for(T_Id y=0; y< mBoxY; y++ )
      for(T_Id z=0; z< mBoxX; z++ )
	if(mLatticeIds->host[met.getCurve().linearizeBoxVectorIndex(x,y,z)] > 0 )
	  countLatticeEntries++;
  assert(nReactiveMonomersChains == countLatticeEntries );  
    mLog( "Check" )
        << "checkReactiveLatticeOccupation: \n"
	<< "nReactiveMonomersChains = " << nReactiveMonomersChains << "\n"
	<< "countLatticeEntries     = " << countLatticeEntries << "\n";
  mPolymerSystemSorted->pop();
  for(T_Id x=0; x< mBoxX; x++ )
    for(T_Id y=0; y< mBoxY; y++ )
      for(T_Id z=0; z< mBoxX; z++ ){
	T_Id LatticeEntry(mLatticeIds->host[met.getCurve().linearizeBoxVectorIndex(x,y,z)]);
	if( LatticeEntry > 0 ){
	  auto r=mPolymerSystemSorted->host[LatticeEntry-1 + mviSubGroupOffsets[1] ];
	  if ( r.x != x || r.y != y || r.z != z  )
	  {
	    std::stringstream error_message;
	    error_message << "LatticeEntry=  "<<LatticeEntry  << " "
			  << "Pos= ("<< x <<"," << y << "," << z << ")" << " "
			  << "mPolymerSystemSorted= ("<< r.x <<"," << r.y << "," << r.z << ")" << "\n";
	    throw std::runtime_error(error_message.str());
	  }
	}
      }
}

/**
 * @brief checks the lattice for possible neighbors
 * @details We randomly choose a direction and look on the lattice for a possibel
 * 	    new partner. In this case the lattice entry is greater one. 
 */
template< typename T_UCoordinateCuda >
__global__ void kernelCheckConnection
(
    typename CudaVec4< T_UCoordinateCuda >::value_type
                const * const __restrict__ dpPolymerSystem          ,
    uint32_t            const              iOffset                  ,
    T_Id              * const              dLatticeIds              ,
    T_Id              * const              dpFlag                   ,
    uint8_t     const * const              dpNeighborsSizesCrossLink,
    uint8_t     const * const              dpNeighborsSizesChain    ,
    T_Id                const              nMonomers                ,
    uint64_t            const              rSeed                    ,
    uint64_t            const              rGlobalIteration         ,
    Method              const              met
){
    uint32_t rn;
    int iGrid = 0;
    for ( uint32_t iMonomer = blockIdx.x * blockDim.x + threadIdx.x;
          iMonomer < nMonomers; iMonomer += gridDim.x * blockDim.x, ++iGrid )
    {
        auto const r0 = dpPolymerSystem[ iOffset + iMonomer ];
	if ( dcCrossLinkMaxNumLinks == dpNeighborsSizesCrossLink[ iMonomer ] ) continue; //already max number of connections for the crosslinker
        if ( iGrid % 1 == 0 ) //for what is this  
        {
	  Saru rng(rGlobalIteration,iMonomer,rSeed);
	  rn =rng.rng32();
        }
        int direction = rn % 6;
	/* select random direction. Do this with bitmasking instead of lookup ??? */
	
        typename CudaVec4< T_UCoordinateCuda >::value_type const r1 = {
            T_UCoordinateCuda( r0.x + DXTable2_d[ direction ] ),
            T_UCoordinateCuda( r0.y + DYTable2_d[ direction ] ),
            T_UCoordinateCuda( r0.z + DZTable2_d[ direction ] ) };

	auto const PartnerlatticeEntry = dLatticeIds[met.getCurve().linearizeBoxVectorIndex(r1.x,r1.y,r1.z )];
// 	printf("ID_X=%d, ID_X=%d , IC_c=%d, (%d,%d,%d), (%d,%d,%d)\n", iOffset + iMonomer, iMonomer, PartnerlatticeEntry-1, r1.x,r1.y,r1.z ,r0.x,r0.y,r0.z);
	//Partner Id start at 1!!!
	if ( PartnerlatticeEntry == 0 ) continue; //is not reactive for 0  or cross link (do not allow connections betweeen cross links)
	if ( dcChainMaxNumLinks == dpNeighborsSizesChain[ PartnerlatticeEntry -1 ] ) continue; //already max number of connections for the chain
        dpFlag[ iMonomer + 1 ] = PartnerlatticeEntry ; 
    }
}
template< typename T_UCoordinateCuda >
void UpdaterGPUScBFM_AB_Connection< T_UCoordinateCuda >::launch_CheckConnection(
  const size_t nBlocks, const size_t nThreads, 
  const size_t iSpeciesCrossLink, const size_t iSpeciesChain, const uint64_t seed )
{
   kernelCheckConnection< T_UCoordinateCuda > 
  <<<nBlocks, nThreads, 0, mStream>>>(                
      mPolymerSystemSorted->gpu,       
      mviSubGroupOffsets[ iSpeciesCrossLink ], 
      mLatticeIds->gpu,
      mCrossLinkFlags,
      mNeighborsSortedSizes->gpu + mviSubGroupOffsets[ iSpeciesCrossLink ], 
      mNeighborsSortedSizes->gpu + mviSubGroupOffsets[ iSpeciesChain ], 
      mnElementsInGroup[ iSpeciesCrossLink ],                       
      seed, 
      hGlobalIterator,                                         
      met
  );
  hGlobalIterator++;
  //reset vectors 
  thrust::sequence(thrust::device, mCrossLinkIDS, mCrossLinkIDS+flagArraySize,0 );
//   Connection connection(nReactiveMonomersCrossLinks);
  connection.resetMultipleIDs(mCrossLinkIDS,mCrossLinkFlags,mStream);
  if( mLog( "Check" ).isActive()){
    mLog( "Check" ) << "Copying the flags and ids for a check:\n" ;
    T_Id * hCrossLinks; 
    hCrossLinks = (T_Id *) malloc(sizeof(T_Id) *flagArraySize);
    T_Id * hCrossLinkFlags; 
    hCrossLinkFlags = (T_Id *) malloc(sizeof(T_Id) *flagArraySize);
    cudaMemcpy(hCrossLinks, mCrossLinkIDS, sizeof(T_Id) *flagArraySize, cudaMemcpyDeviceToHost );
    cudaMemcpy(hCrossLinkFlags, mCrossLinkFlags, sizeof(T_Id) *flagArraySize, cudaMemcpyDeviceToHost );
    miNewToi->pop();
    for( size_t i =0; i < flagArraySize ;i++)
    {
      auto const  r0(mPolymerSystemSorted->host[hCrossLinks[i]-1+mviSubGroupOffsets[ 0 ]]);
      auto const  r1(mPolymerSystemSorted->host[hCrossLinkFlags[i]-1+mviSubGroupOffsets[ 1 ]]);
      if (hCrossLinkFlags[i]>0)
	mLog("Check") << "ID= " << miNewToi->host[ hCrossLinks[i]-1 ]<< " Flags= " << miNewToi->host[ hCrossLinkFlags[i]-1 ] <<"\n";
    }
  }
}

template< typename T_UCoordinateCuda >
__global__ void kernelApplyConnection
(
    T_Id              * const              mCrossLinkFlags         ,
    T_Id              * const              mCrossLinkIDS           ,
    T_Id                const              flagArraySize           ,               
    T_Id              * const              dpNeighborsMonomer      ,
    T_Id              * const              dpNeighborsPartner      ,
    uint32_t            const              rNeighborsPitchElementsMonomer ,
    uint32_t            const              rNeighborsPitchElementsPartner ,
    uint8_t           * const              dpNeighborsSizesMonomer ,
    uint8_t           * const              dpNeighborsSizesPartner ,
    uint32_t 		const 		   iOffsetCrossLinks       ,
    uint32_t 		const 		   iOffsetChains
){
    for ( auto i = blockIdx.x * blockDim.x + threadIdx.x;
          i < flagArraySize; i += gridDim.x * blockDim.x )
    {
      auto iPartner(mCrossLinkFlags[i]);
      auto iMonomer(mCrossLinkIDS[i]);
      if (iPartner == 0 || iMonomer == 0 ) 
      	continue; //no Partner found -> go to next Crosslink in the grid 
      
      iPartner--;
      iMonomer--;
      dpNeighborsMonomer[ dpNeighborsSizesMonomer[ iMonomer ] * rNeighborsPitchElementsMonomer + iMonomer ] = iOffsetChains + iPartner; 
      dpNeighborsPartner[ dpNeighborsSizesPartner[ iPartner ] * rNeighborsPitchElementsPartner + iPartner ] = iOffsetCrossLinks + iMonomer; 
      dpNeighborsSizesMonomer[ iMonomer ]++;
      dpNeighborsSizesPartner[ iPartner ]++; 
//       printf(" %d Connect monomers: %d with %d , %d ,%d ,%d ,%d \n", i,iMonomer, iPartner, 
// 	     dpNeighborsMonomer[ (dpNeighborsSizesMonomer[ iMonomer ]-1) * rNeighborsPitchElementsMonomer + iMonomer ], 
// 	     dpNeighborsPartner[ (dpNeighborsSizesPartner[ iPartner ]-1) * rNeighborsPitchElementsPartner + iPartner ],
// 	     dpNeighborsSizesMonomer[ iMonomer ],
// 	     dpNeighborsSizesPartner[ iPartner ]
// 	    ); 
    }
}


template< typename T_UCoordinateCuda >
void UpdaterGPUScBFM_AB_Connection< T_UCoordinateCuda >::launch_ApplyConnection(
  const size_t nBlocks , const size_t   nThreads, 
  const size_t MonomerSpecies,
  const size_t PartnerSpecies
)
{ 
  kernelApplyConnection<T_UCoordinateCuda><<<nBlocks,nThreads,0,mStream>>>(
    mCrossLinkFlags,
    mCrossLinkIDS,
    flagArraySize, 
    mNeighborsSorted->gpu + mNeighborsSortedInfo.getMatrixOffsetElements( MonomerSpecies ), 
    mNeighborsSorted->gpu + mNeighborsSortedInfo.getMatrixOffsetElements( PartnerSpecies ), 
    mNeighborsSortedInfo.getMatrixPitchElements( MonomerSpecies ),
    mNeighborsSortedInfo.getMatrixPitchElements( PartnerSpecies ),       
    mNeighborsSortedSizes->gpu + mviSubGroupOffsets[ MonomerSpecies ],
    mNeighborsSortedSizes->gpu + mviSubGroupOffsets[ PartnerSpecies ],
    mviSubGroupOffsets[ MonomerSpecies ],   
    mviSubGroupOffsets[ PartnerSpecies ]
  );
  tracker.trackConnections( mCrossLinkIDS, mCrossLinkFlags, flagArraySize, miNewToi->gpu, mviSubGroupOffsets[ MonomerSpecies ], mviSubGroupOffsets[ PartnerSpecies ], mAge );
}



template< typename T_UCoordinateCuda > 
UpdaterGPUScBFM_AB_Connection<T_UCoordinateCuda>::UpdaterGPUScBFM_AB_Connection():
BaseClass()                         , 
mLatticeIds                 ( NULL ),
mCrossLinkFlags             ( NULL ),
mCrossLinkIDS               ( NULL ),
nReactiveMonomers           ( 0    ),
nReactiveMonomersChains     ( 0    ),
nReactiveMonomersCrossLinks ( 0    ),
crosslinkFunctionality      ( 0    )
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
void UpdaterGPUScBFM_AB_Connection<T_UCoordinateCuda>::destruct(){
      
    DeleteMirroredObject deletePointer;
    deletePointer( mLatticeIds       , "mLatticeIds"        );
//     CUDA_ERROR(cudaFree(mCrossLinkFlags));  
//     CUDA_ERROR(cudaFree(mCrossLinkIDS)); 
    if ( deletePointer.nBytesFreed > 0 )
    {
        mLog( "Info" )
            << "Freed a total of "
            << prettyPrintBytes( deletePointer.nBytesFreed )
            << " on GPU and host RAM.\n";
    }
}
template< typename T_UCoordinateCuda > 
UpdaterGPUScBFM_AB_Connection<T_UCoordinateCuda>::~UpdaterGPUScBFM_AB_Connection()
{
  this->destruct();    
  destruct();
}

template< typename T_UCoordinateCuda >
void UpdaterGPUScBFM_AB_Connection<T_UCoordinateCuda>::cleanup()
{
    this->destruct();    
    destruct();
    cudaDeviceSynchronize();
    cudaProfilerStop();
    
}

template < typename T_UCoordinateCuda >
void UpdaterGPUScBFM_AB_Connection<T_UCoordinateCuda>::initialize()
{
  

  BaseClass::setAutoColoring(false);
  mLog( "Info" )<< "Start manual coloring of the graph...\n" ;
  //do manual coloring 
  for ( auto i = 0; i < mnAllMonomers ; i++)
  {
    T_Id color(( i % 2)==0 ? 2 :3);
    mGroupIds.push_back(color ); 
  }

  for (auto i = 0; i < nReactiveMonomers; i++)
  {
    mGroupIds[mNewToOldReactiveID[i]] = (mMonomerReactivity[i].maxNumLinks == 2 ) ? 1 : 0 ;
    if (i <20 ) 
      mLog( "Info" )<< "mGroups[" << mNewToOldReactiveID[i] << "]= "<< mGroupIds[mNewToOldReactiveID[i]] <<"\n" ;
  }
  mLog( "Info" )<< "Start manual coloring of the graph...done\n" ;
  mLog( "Info" )<< "Initialize baseclass \n" ;
  BaseClass::initialize();

  mLog( "Info" )<<"Cross link functionality is "<< crosslinkFunctionality << "\n";
  CUDA_ERROR( cudaMemcpyToSymbol( dcCrossLinkMaxNumLinks, &crosslinkFunctionality, sizeof( crosslinkFunctionality ) ) );
  flagArraySize = (4*ceil((nReactiveMonomersCrossLinks+1)*1.0/4.) );
  CUDA_ERROR(cudaMalloc((void **) &mCrossLinkIDS, sizeof(T_Id)*flagArraySize));
  CUDA_ERROR(cudaMalloc((void **) &mCrossLinkFlags, sizeof(T_Id)*flagArraySize));

  { decltype( dcBoxX      ) x = mBoxX     ; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxX     , &x, sizeof(x) ) ); }
  { decltype( dcBoxY      ) x = mBoxY     ; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxY     , &x, sizeof(x) ) ); }
  { decltype( dcBoxZ      ) x = mBoxZ     ; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxZ     , &x, sizeof(x) ) ); }
  { decltype( dcBoxXM1    ) x = mBoxXM1   ; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxXM1   , &x, sizeof(x) ) ); }
  { decltype( dcBoxYM1    ) x = mBoxYM1   ; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxYM1   , &x, sizeof(x) ) ); }
  { decltype( dcBoxZM1    ) x = mBoxZM1   ; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxZM1   , &x, sizeof(x) ) ); }
  { decltype( dcBoxXLog2  ) x = mBoxXLog2 ; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxXLog2 , &x, sizeof(x) ) ); }
  { decltype( dcBoxXYLog2 ) x = mBoxXYLog2; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxXYLog2, &x, sizeof(x) ) ); }
  uint32_t tmp_DXTable2[6] = { 0u-2u,2,  0,0,  0,0 };
  uint32_t tmp_DYTable2[6] = {  0,0, 0u-2u,2,  0,0 };
  uint32_t tmp_DZTable2[6] = {  0,0,  0,0, 0u-2u,2 };
  CUDA_ERROR( cudaMemcpyToSymbol( DXTable2_d, tmp_DXTable2, sizeof( tmp_DXTable2 ) ) ); 
  CUDA_ERROR( cudaMemcpyToSymbol( DYTable2_d, tmp_DYTable2, sizeof( tmp_DXTable2 ) ) );
  CUDA_ERROR( cudaMemcpyToSymbol( DZTable2_d, tmp_DZTable2, sizeof( tmp_DXTable2 ) ) );
  if (mLog( "Info" ).isActive()){
    mLog( "Info" )<< "Check box dimensions \n" ;
    CheckBoxDimensions<<<1,1,0,mStream>>>();
    CUDA_ERROR( cudaStreamSynchronize( mStream ) ); // finish e.g. initializations
  }
  mLog( "Info" )<< "Initialize baseclass.done. \n" ;
  CrossLinkSpecies = 0; 
  ChainEndSpecies  = 1; 
  initializeReactiveLattice();
  mLog( "Info" )<< "Initialize lattice.done. \n" ;
  tracker.init(100, nReactiveMonomersCrossLinks+1, mStream);
  mLog( "Info" ) << "nReactiveMonomersCrossLinks = " << nReactiveMonomersCrossLinks+1 <<"\n";
  connection.setArraySize(nReactiveMonomersCrossLinks);
  connection.init();
  miNewToi->popAsync();
  CUDA_ERROR( cudaStreamSynchronize( mStream ) ); // finish e.g. initializations
  tracker.setIDOffset(miNewToi->host[mviSubGroupOffsets[ CrossLinkSpecies ]]);

}

template< typename T_UCoordinateCuda >
void UpdaterGPUScBFM_AB_Connection<T_UCoordinateCuda>::setNrOfReactiveMonomers( T_Id nReactiveMonomers_ , T_Id nReactiveMonomersCrossLinks_, T_Id nReactiveMonomersChains_ )
{
    if ( nReactiveMonomers != 0 || nReactiveMonomersCrossLinks != 0 || nReactiveMonomersChains != 0 )
    {
        std::stringstream msg;
        msg << "[" << __FILENAME__ << "::setNrOfReactiveMonomers] "
            << "Number of nReactiveMonomers already set to           " << nReactiveMonomers << "!\n"
	    << "Number of nReactiveMonomersChains already set to     " << nReactiveMonomersChains << "!\n"
	    << "Number of nReactiveMonomersCrossLinks already set to " << nReactiveMonomersCrossLinks << "!\n";
        throw std::runtime_error( msg.str() );
    }
    nReactiveMonomers           = nReactiveMonomers_;
    nReactiveMonomersChains     = nReactiveMonomersChains_;
    nReactiveMonomersCrossLinks = nReactiveMonomersCrossLinks_;
    mLog( "Info" )
	  << "Nr of reactive monomers   "<< nReactiveMonomers <<"\n" 
	  << "Nr of reactive crosslinks "<< nReactiveMonomersCrossLinks <<"\n" 
	  << "Nr of reactive chain ends "<< nReactiveMonomersChains <<"\n" ;
    
};

template< typename T_UCoordinateCuda >
void UpdaterGPUScBFM_AB_Connection<T_UCoordinateCuda>::setReactiveGroup(T_Id monID_, bool reactivity_, T_MaxNumLinks maxNumLinks_){

  
  //fill  mMonomerReactivity
  if (reactivity_ ){
    mNewToOldReactiveID.push_back(monID_);
    D_MonomerReactivity monReact;
    monReact.reactivity=reactivity_;
    monReact.maxNumLinks=maxNumLinks_;
    mMonomerReactivity.push_back(monReact);
    if(maxNumLinks_ > crosslinkFunctionality) crosslinkFunctionality = maxNumLinks_;
  }
  
}

template< typename T_UCoordinateCuda >
void UpdaterGPUScBFM_AB_Connection<T_UCoordinateCuda>::initializeReactiveLattice()
{
 if ( mLatticeIds != NULL )
    {
        std::stringstream msg;
        msg << "[" << __FILENAME__ << "::initializeReactiveLattice] "
            << "Initialize was already called and may not be called again "
            << "until cleanup was called!";
        mLog( "Error" ) << msg.str();
        throw std::runtime_error( msg.str() );
    }
    size_t nBytesLatticeTmp = mBoxX * mBoxY * mBoxZ * sizeof(T_Id);
     mLog( "Info" ) << "Allocate "<< nBytesLatticeTmp/1024<<"kB  memory for lattice \n";  
    mLatticeIds  = new MirroredTexture< T_Id >( nBytesLatticeTmp, mStream );
}


template< typename T_UCoordinateCuda  >
void UpdaterGPUScBFM_AB_Connection< T_UCoordinateCuda >::runSimulationOnGPU
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
// 	tracker.setAge(mAge);
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
	    if (!diagMovesOn)  
	    {
		this-> template launch_CheckSpecies<6>(nBlocks, nThreads, iSpecies, iOffsetLatticeTmp, seed);
		if ( useCudaMemset )
		    launch_PerformSpeciesAndApply(nBlocks, nThreads, iSpecies, texLatticeTmp);
		else
		    launch_PerformSpecies(nBlocks,nThreads,iSpecies,texLatticeTmp);
	    }else 
	    {
		this-> template launch_CheckSpecies<18>(nBlocks, nThreads, iSpecies, iOffsetLatticeTmp, seed);
		if ( useCudaMemset )
		    launch_PerformSpeciesAndApply(nBlocks, nThreads, iSpecies, texLatticeTmp);
		else
		    launch_PerformSpecies(nBlocks,nThreads,iSpecies,texLatticeTmp);
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
	
    } // iStep
    
    std::clock_t const t1 = std::clock();
    double const dt = float(t1-t0) / CLOCKS_PER_SEC;
    mLog( "Info" )
    << "run time (GPU): " << nMonteCarloSteps << "\n"
    << "mcs = " << nMonteCarloSteps  << "  speed [performed monomer try and move/s] = MCS*N/t: "
    << nMonteCarloSteps * ( mnAllMonomers / dt )  << "     runtime[s]:" << dt << "\n";
    doCopyBack();
    checkSystem(); // no-op if "Check"-level deactivated
    tracker.dumpReactions();
}

template< typename T_UCoordinateCuda >
void UpdaterGPUScBFM_AB_Connection< T_UCoordinateCuda >::doCopyBack()
{
    mLog( "Stats" ) << "UpdaterGPUScBFM_AB_Connection< T_UCoordinateCuda >::doCopyBackConnectivity() \n";
    doCopyBackMonomerPositions();
    mLog( "Stats" ) << "UpdaterGPUScBFM_AB_Connection< T_UCoordinateCuda >::doCopyBackConnectivity() \n";
    doCopyBackConnectivity(); // -> need to write a kernel for that. its pretty slow!!! (but works :-) )
}

template< typename T_UCoordinateCuda >
void UpdaterGPUScBFM_AB_Connection< T_UCoordinateCuda >::checkBonds() const
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
        dx %= mBoxX; if ( dx < -int( mBoxX )/ 2 ) dx += mBoxX; if ( dx > (int) mBoxX / 2 ) dx -= mBoxX;
        dy %= mBoxY; if ( dy < -int( mBoxY )/ 2 ) dy += mBoxY; if ( dy > (int) mBoxY / 2 ) dy -= mBoxY;
        dz %= mBoxZ; if ( dz < -int( mBoxZ )/ 2 ) dz += mBoxZ; if ( dz > (int) mBoxZ / 2 ) dz -= mBoxZ;
        int erroneousAxis = -1;
        if ( ! ( -3 <= dx && dx <= 3 ) ) erroneousAxis = 0;
        if ( ! ( -3 <= dy && dy <= 3 ) ) erroneousAxis = 1;
        if ( ! ( -3 <= dz && dz <= 3 ) ) erroneousAxis = 2;
	if ( ! ( dx*dx+dy*dy+dz*dz > 10 ) ) erroneousAxis = 3;
        if ( erroneousAxis >= 0 || checkBondVector( dx, dy, dz )  )
        {
            std::stringstream msg;
            msg << "[" << __FILENAME__ << "::checkSystem] ";
            if ( erroneousAxis > 0 && erroneousAxis < 3 )
                msg << "Invalid " << char( 'X' + erroneousAxis ) << "-Bond: ";
	    if ( erroneousAxis == 3 )
	        msg << "Invalid square length=" << dx*dx+dy*dy+dz*dz << ": ";
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

template< typename T_UCoordinateCuda >
void UpdaterGPUScBFM_AB_Connection< T_UCoordinateCuda >::checkSystem() const
{
    if ( ! mLog.isActive( "Check" ) )
        return;
    this -> checkLatticeOccupation();
//     BaseClass::checkBonds();
    for (auto i = 0; i < nReactiveMonomers; i++)
    {
      if (mGroupIds[mNewToOldReactiveID[i]] == 0 )
      {
	if (mNeighbors->host[i].size > crosslinkFunctionality)
	{
	  std::stringstream error_message;
	  error_message << "Exceeds the maximum number of bonds of " <<crosslinkFunctionality << "for crossLinks at monomer Id "
		        <<  i << " with " << mNeighbors->host[i].size << "\n";
	  for (size_t j =0 ; j < mNeighbors->host[i].size; j++ )
	    error_message <<"Neighbor[" <<j << "]= " <<  mNeighbors->host[i].neighborIds[j] << "\n";
	  throw std::runtime_error(error_message.str());
	}
      }
      else if (mGroupIds[mNewToOldReactiveID[i]] == 0 )
      { 
	if (mNeighbors->host[i].size > 2)
	{
	  std::stringstream error_message;
	  error_message << "Exceeds the maximum number of bonds of " << 2 << " for chain ends at monomer Id "
		        <<  i << " with " << mNeighbors->host[i].size << "\n";
	  for (size_t j =0 ; j < mNeighbors->host[i].size; j++ )
	    error_message <<"Neighbor[" <<j << "]= " <<  mNeighbors->host[i].neighborIds[j] << "\n";
	  throw std::runtime_error(error_message.str());
	}
      }
      
    }
    checkBonds();
}

template class UpdaterGPUScBFM_AB_Connection< uint8_t  >;
template class UpdaterGPUScBFM_AB_Connection< uint16_t >;
template class UpdaterGPUScBFM_AB_Connection< uint32_t >;
template class UpdaterGPUScBFM_AB_Connection<  int16_t >;
template class UpdaterGPUScBFM_AB_Connection<  int32_t >;