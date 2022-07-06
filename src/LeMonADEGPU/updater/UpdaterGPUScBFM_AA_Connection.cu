/*--------------------------------------------------------------------------------
    ooo      L   attice-based  |
  o\.|./o    e   xtensible     | LeMonADE: An Open Source Implementation of the
 o\.\|/./o   Mon te-Carlo      |           Bond-Fluctuation-Model for Polymers
oo--GPU--oo  A   lgorithm and  |
 o/./|\.\o   D   evelopment    | Copyright (C) 2013-2015 by
  o/.|.\o    E   nvironment    | LeMonADE Principal Developers (see AUTHORS)
    ooo                        |
----------------------------------------------------------------------------------

This file is part of LeMonADEGPU.

LeMonADE is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

LeMonADE is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with LeMonADE.  If not, see <http://www.gnu.org/licenses/>.

--------------------------------------------------------------------------------*/

/*
 * UpdaterGPUScBFM_AA_Connection.cu
 *
 *  Created on: 27.06.2019
 *      Authors: Toni Mueller
 */

#include <LeMonADEGPU/updater/UpdaterGPUScBFM_AA_Connection.h>
// #include <LeMonADEGPU/updater/UpdaterGPUScBFM.h>
#include <LeMonADEGPU/utility/cudacommon.hpp>
#include <LeMonADEGPU/core/Method.h>
#include <LeMonADEGPU/utility/DeleteMirroredObject.h>
#include <cuda_profiler_api.h>              // cudaProfilerStop
#include <LeMonADEGPU/utility/AutomaticThreadChooser.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>

#include <extern/Fundamental/BitsCompileTime.hpp>
#include <LeMonADEGPU/utility/cudacommon.hpp>

#include <LeMonADEGPU/utility/SelectiveLogger.hpp>
#include <LeMonADEGPU/utility/graphColoring.tpp>
#include <LeMonADEGPU/core/rngs/Saru.h>
#include <LeMonADEGPU/core/MonomerEdges.h>
#include <LeMonADEGPU/core/constants.cuh>
#include <LeMonADEGPU/feature/BoxCheck.h>
#include <LeMonADEGPU/core/Method.h>

#include <LeMonADEGPU/utility/MirroredVector.h>
#include <LeMonADEGPU/utility/DeleteMirroredObject.h>
#include <LeMonADEGPU/core/BondVectorSet.h>
#include <LeMonADEGPU/core/kernelConnection.h>
#include <LeMonADEGPU/utility/GPUConnectionTracker.h>

using T_Flags            = UpdaterGPUScBFM_AA_Connection< uint8_t >::T_Flags      ;
using T_Lattice          = UpdaterGPUScBFM< uint8_t >::T_Lattice    ;
using T_Id               = UpdaterGPUScBFM< uint8_t >::T_Id         ;
using getBitPackedTextureFunction = UpdaterGPUScBFM<uint8_t>::getBitPackedTextureFunction;
__device__ getBitPackedTextureFunction functor = &BitPacking::bitPackedTextureGetStandard;
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
void UpdaterGPUScBFM_AA_Connection< T_UCoordinateCuda >::launch_initializeReactiveLattice(
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
void UpdaterGPUScBFM_AA_Connection< T_UCoordinateCuda >::launch_resetReactiveLattice(
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
void UpdaterGPUScBFM_AA_Connection< T_UCoordinateCuda >::checkReactiveLatticeOccupation()  
{
  mLatticeIds->pop(0);
  //check the number of reactive lattice entries 
  uint32_t countLatticeEntries(0);
  for(T_Id x=0; x< mBoxX; x++ )
    for(T_Id y=0; y< mBoxY; y++ )
      for(T_Id z=0; z< mBoxX; z++ )
	if(mLatticeIds->host[met.getCurve().linearizeBoxVectorIndex(x,y,z)] > 0 )
	  countLatticeEntries++;
  assert(nReactiveMonomers == countLatticeEntries );  
  
  mLog( "Check" )
        << "checkReactiveLatticeOccupation: \n"
	<< "nReactiveMonomers   = " << nReactiveMonomers << "\n"
	<< "countLatticeEntries = " << countLatticeEntries << "\n";
  //check if the lattice entry is on the right place 
  mPolymerSystemSorted->pop();
  for(T_Id x=0; x< mBoxX; x++ )
    for(T_Id y=0; y< mBoxY; y++ )
      for(T_Id z=0; z< mBoxX; z++ )
      {
	T_Id LatticeEntry(mLatticeIds->host[met.getCurve().linearizeBoxVectorIndex(x,y,z)]);
	if( LatticeEntry > 0 ){
// 	  auto r=mPolymerSystemSorted->host[LatticeEntry-1 + mviSubGroupOffsets[1] ];
	  auto r=mPolymerSystemSorted->host[LatticeEntry-1 + mviSubGroupOffsets[0] ];
	  if ( r.x % mBoxX != x || r.y % mBoxY != y || r.z % mBoxZ != z  )
	  {
	    std::stringstream error_message;
	    error_message << "LatticeEntry="<<LatticeEntry  << " "
			  << "Pos=("<< x <<"," << y << "," << z << ")" << " "
			  << "mPolymerSystemSorted=("<< (uint32_t)r.x % mBoxX <<"," << (uint32_t)r.y% mBoxY << "," << (uint32_t)r.z % mBoxZ<< ")" << "\n";
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
    uint8_t     const * const              dpNeighborsSizes         ,
    T_Id                const              nMonomers                ,
    uint64_t            const              rSeed                    ,
    uint64_t            const              rGlobalIteration         ,
    BoxCheck                               bCheck, 
    Method              const              met
){
    uint32_t rn;
    int iGrid = 0;
    for ( uint32_t iMonomer = blockIdx.x * blockDim.x + threadIdx.x;
          iMonomer < nMonomers; iMonomer += gridDim.x * blockDim.x, ++iGrid )
    {
	dpFlag[ iMonomer + 1 ] =0;
        auto const r0 = dpPolymerSystem[ iOffset + iMonomer ];
	if ( dcChainMaxNumLinks == dpNeighborsSizes[ iMonomer ] ) continue; //already max number of connections for the crosslinker
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
//-> need a statement to check wheter the new connection would cross the box
        // otherwise a connection could establish across the box for nonperiodic boundary conditions
        if ( bCheck(r1.x,r1.y,r1.z) ) continue; 
	auto const PartnerlatticeEntry = dLatticeIds[met.getCurve().linearizeBoxVectorIndex(r1.x,r1.y,r1.z )];
	//Partner Id start at 1!!!
	if ( PartnerlatticeEntry == 0 ) continue; //is not reactive for 0  or cross link (do not allow connections betweeen cross links)
	if ( dcChainMaxNumLinks == dpNeighborsSizes[ PartnerlatticeEntry -1 ] ) continue; //already max number of connections for the chain
// 	printf("ID_X=%d, ID_X=%d , IC_c=%d, (%d,%d,%d), (%d,%d,%d)\n", iOffset + iMonomer, iMonomer, PartnerlatticeEntry-1, r1.x,r1.y,r1.z ,r0.x,r0.y,r0.z);
        dpFlag[ iMonomer + 1 ] = PartnerlatticeEntry ; 
    }
}
template< typename T_UCoordinateCuda >
void UpdaterGPUScBFM_AA_Connection< T_UCoordinateCuda >::launch_CheckConnection(
  const size_t nBlocks, const size_t nThreads, const size_t iSpecies, const uint64_t seed )
{
   kernelCheckConnection< T_UCoordinateCuda > 
  <<<nBlocks, nThreads, 0, mStream>>>(                
      mPolymerSystemSorted->gpu,       
      mviSubGroupOffsets[ iSpecies ], 
      mLatticeIds->gpu,
      mChainEndFlags,
      mNeighborsSortedSizes->gpu + mviSubGroupOffsets[ iSpecies ], 
      mnElementsInGroup[ iSpecies ],                       
      seed, 
      hGlobalIterator, 
      boxCheck,                                               
      met
  );
  hGlobalIterator++;
  //reset vectors 
  thrust::sequence(thrust::device, mChainEndIDS, mChainEndIDS+flagArraySize,0 );
//   Connection connection(nReactiveMonomersCrossLinks);
  connection.resetMultipleIDs(mChainEndIDS,mChainEndFlags,mStream);
  CUDA_ERROR( cudaStreamSynchronize( mStream ) );
  connection.resetMultipleBonds(mChainEndIDS,mChainEndFlags,mStream);
  if( mLog( "Check" ).isActive()){
    mLog( "Check" ) << "Copying the flags and ids for a check:\n" ;
    T_Id * hCrossLinks; 
    hCrossLinks = (T_Id *) malloc(sizeof(T_Id) *flagArraySize);
    T_Id * hCrossLinkFlags; 
    hCrossLinkFlags = (T_Id *) malloc(sizeof(T_Id) *flagArraySize);
    cudaMemcpy(hCrossLinks, mChainEndIDS, sizeof(T_Id) *flagArraySize, cudaMemcpyDeviceToHost );
    cudaMemcpy(hCrossLinkFlags, mChainEndFlags, sizeof(T_Id) *flagArraySize, cudaMemcpyDeviceToHost );
    miNewToi->pop();
    for( size_t i =0; i < flagArraySize ;i++)
    {
      auto const  r0(mPolymerSystemSorted->host[ hCrossLinks[i]    -1+mviSubGroupOffsets[ iSpecies ] ]);
      auto const  r1(mPolymerSystemSorted->host[ hCrossLinkFlags[i]-1+mviSubGroupOffsets[ iSpecies ] ]);
      if (hCrossLinkFlags[i]>0)
	mLog("Check") << "ID= " << miNewToi->host[ hCrossLinks[i]-1 ]<< " Flags= " << miNewToi->host[ hCrossLinkFlags[i]-1 ] << " ID_new=" << hCrossLinks[i]-1 << " Flag_new="<< hCrossLinkFlags[i]-1 << "\n";
    }
  }
}

template< typename T_UCoordinateCuda >
__global__ void kernelApplyConnection
(
    T_Id              * const mChainEndFlags         ,
    T_Id              * const mChainEndIDS           ,
    T_Id                const flagArraySize          ,               
    T_Id              * const dpNeighbors            ,
    uint32_t            const rNeighborsPitchElements,
    uint8_t           * const dpNeighborsSizes       ,
    uint32_t 		const iOffset                ,
    uint8_t           * const MoveFlags
){
    for ( auto i = blockIdx.x * blockDim.x + threadIdx.x;
          i < flagArraySize; i += gridDim.x * blockDim.x )
    {
      auto iPartner(mChainEndFlags[i]);
      auto iMonomer(mChainEndIDS[i]);
      if (iPartner == 0 || iMonomer == 0 ) 
      	continue; //no Partner found -> go to next Crosslink in the grid 
      
      iPartner--;
      iMonomer--;
      dpNeighbors[ dpNeighborsSizes[ iMonomer ] * rNeighborsPitchElements + iMonomer ] = iOffset + iPartner; 
      dpNeighbors[ dpNeighborsSizes[ iPartner ] * rNeighborsPitchElements + iPartner ] = iOffset + iMonomer; 
      dpNeighborsSizes[ iMonomer ]++;
      dpNeighborsSizes[ iPartner ]++; 
      MoveFlags[iMonomer] = 1;
      MoveFlags[iPartner] = 0;
//       printf("ConnectAA: %d Connect monomers: =%d with =%d , n1=%d, n2=%d ,s1=%d ,s2=%d, flag1=%d, flags2=%d \n", i,iMonomer, iPartner, 
// 	     dpNeighbors[ (dpNeighborsSizes[ iMonomer ]-1) * rNeighborsPitchElements + iMonomer ], 
// 	     dpNeighbors[ (dpNeighborsSizes[ iPartner ]-1) * rNeighborsPitchElements + iPartner ],
// 	     dpNeighborsSizes[ iMonomer ],
// 	     dpNeighborsSizes[ iPartner ],
// 	     MoveFlags[iMonomer],MoveFlags[iPartner]
// 	    ); 
    }
}

template< typename T_UCoordinateCuda >
void UpdaterGPUScBFM_AA_Connection< T_UCoordinateCuda >::launch_ApplyConnection(
  const size_t nBlocks , const size_t   nThreads, 
  const size_t iSpecies
)
{ 
  
  kernelApplyConnection<T_UCoordinateCuda><<<nBlocks,nThreads,0,mStream>>>(
    mChainEndFlags,
    mChainEndIDS,
    flagArraySize, 
    mNeighborsSorted->gpu + mNeighborsSortedInfo.getMatrixOffsetElements( iSpecies ), 
    mNeighborsSortedInfo.getMatrixPitchElements( iSpecies ),
    mNeighborsSortedSizes->gpu + mviSubGroupOffsets[ iSpecies ],
    mviSubGroupOffsets[ iSpecies ],
    AAMonomerFlag->gpu
  );
  tracker.trackConnections( mChainEndFlags, mChainEndIDS, flagArraySize, miNewToi->gpu,miToiNew->gpu,mviSubGroupOffsets[ iSpecies ],mviSubGroupOffsets[ iSpecies ], mAge, mPolymerSystemSorted, mviPolymerSystemSortedVirtualBox );
  CUDA_ERROR(cudaDeviceSynchronize());
  CUDA_ERROR( cudaStreamSynchronize( mStream ) );
}



template< typename T_UCoordinateCuda > 
UpdaterGPUScBFM_AA_Connection<T_UCoordinateCuda>::UpdaterGPUScBFM_AA_Connection():
BaseClass()                         , 
mLatticeIds                 ( NULL ),
mChainEndFlags              ( NULL ),
mChainEndIDS                ( NULL ),
AAMonomerFlag               ( NULL ),
nReactiveMonomers           ( 0    ),
crosslinkFunctionality      ( 4    )
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
void UpdaterGPUScBFM_AA_Connection<T_UCoordinateCuda>::destruct(){
      
    DeleteMirroredObject deletePointer;
    deletePointer( mLatticeIds   , "mLatticeIds"  );
    deletePointer( AAMonomerFlag , "AAMonomerFlag");
//     CUDA_ERROR(cudaFree(mChainEndFlags));  
//     CUDA_ERROR(cudaFree(mChainEndIDS)); 
    if ( deletePointer.nBytesFreed > 0 )
    {
        mLog( "Info" )
            << "Freed a total of "
            << prettyPrintBytes( deletePointer.nBytesFreed )
            << " on GPU and host RAM.\n";
    }
}
template< typename T_UCoordinateCuda > 
UpdaterGPUScBFM_AA_Connection<T_UCoordinateCuda>::~UpdaterGPUScBFM_AA_Connection()
{
  this->destruct();    
  destruct();
}

template< typename T_UCoordinateCuda >
void UpdaterGPUScBFM_AA_Connection<T_UCoordinateCuda>::cleanup()
{
    tracker.dumpReactions();
    this->destruct();    
    destruct();   
    cudaDeviceSynchronize();
    cudaProfilerStop();
    
}

template < typename T_UCoordinateCuda >
void UpdaterGPUScBFM_AA_Connection<T_UCoordinateCuda>::initialize()
{
  

  BaseClass::setAutoColoring(false);
  mLog( "Info" )<< "Start manual coloring of the graph...\n" ;
  //do manual coloring
  //I assume a star like structure where the end groups of the stars are the reactive monomers.
  //To achieve a good labeling we start at the reactive end groups and reduce the ID by one 
  //and increase the color by one as long as there are only two connections. If there are more 
  //than two, then this monomer is also colored and then we continue with the next reactive end 
  //group.
  /*  monomer ids      coloring
   *      7                0
   *      |                |
   *      6                1
   *      |                |
   *  5-4-1-2-3  -->   0-1-2-1-0
   *      |                |
   *      8                1
   *      |                |
   *      9                0
   */
  mGroupIds.resize(mnAllMonomers,0);
  int32_t armLength(mNewToOldReactiveID[0]);
  int32_t functionality(mNeighbors->host[ 0 ].size);
  int32_t DeltaID(0);
  for (uint32_t i =0 ; i < mnAllMonomers; i ++ )
  {
    if ( i % (armLength*functionality+1) == 0 )
    {
      mGroupIds[i]=3;
      if(i %2 == 0 || armLength % 2 == 1 ) DeltaID=0; //even star number  or for uneven number of additional monomers attachted to the center monomer
      else DeltaID=1; //odd star number
    }
    else
    {
      i%2 == 1 ? mGroupIds[i]=1+DeltaID  :  mGroupIds[i]=2-DeltaID;
    }
  }
  for (auto i = 0; i < nReactiveMonomers; i++)
    mGroupIds[mNewToOldReactiveID[i]] = 0 ;
  for (uint32_t i =0 ; i < mnAllMonomers; i ++ )
      if (i < 20 ) 
	  mLog( "Info" )<< "mGroups[" << i << "]= "<< mGroupIds[i] << "\n" ;
  
  mLog( "Info" )<< "Start manual coloring of the graph...done\n" ;
  mLog( "Info" )<< "Initialize baseclass \n" ;
  BaseClass::initialize();

  mLog( "Info" )<<"Cross link functionality is "<< crosslinkFunctionality << "\n";
  
  flagArraySize = (4*ceil((nReactiveMonomers+1)*1.0/4.) );
  CUDA_ERROR(cudaMalloc((void **) &mChainEndIDS, sizeof(T_Id)*flagArraySize));
  CUDA_ERROR(cudaMalloc((void **) &mChainEndFlags, sizeof(T_Id)*flagArraySize));

  { decltype( dcBoxX      ) x = mBoxX     ; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxX     , &x, sizeof(x) ) ); }
  { decltype( dcBoxY      ) x = mBoxY     ; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxY     , &x, sizeof(x) ) ); }
  { decltype( dcBoxZ      ) x = mBoxZ     ; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxZ     , &x, sizeof(x) ) ); }
  { decltype( dcBoxXM1    ) x = mBoxXM1   ; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxXM1   , &x, sizeof(x) ) ); }
  { decltype( dcBoxYM1    ) x = mBoxYM1   ; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxYM1   , &x, sizeof(x) ) ); }
  { decltype( dcBoxZM1    ) x = mBoxZM1   ; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxZM1   , &x, sizeof(x) ) ); }
//   { decltype( dcBoxXLog2  ) x = mBoxXLog2 ; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxXLog2 , &x, sizeof(x) ) ); }
//   { decltype( dcBoxXYLog2 ) x = mBoxXYLog2; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxXYLog2, &x, sizeof(x) ) ); }
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
//   CrossLinkSpecies = 0; 
  ChainEndSpecies  = 0; 
  initializeReactiveLattice();
  mLog( "Info" )<< "Initialize lattice.done. \n" ;
  tracker.init(100, nReactiveMonomers+1, mStream, mBoxX, mBoxY,mBoxZ);
  mLog( "Info" ) << "nReactiveMonomers = " << nReactiveMonomers+1 <<"\n";
  connection.setArraySize(nReactiveMonomers);
  connection.init();
  miNewToi->popAsync();
  CUDA_ERROR( cudaStreamSynchronize( mStream ) ); // finish e.g. initializations
//   tracker.setIDOffset(miNewToi->host[mviSubGroupOffsets[ ChainEndSpecies ]]);

  //initialize the flags for the reactive monomers 
  assert( AAMonomerFlag      == NULL );
  AAMonomerFlag = new MirroredTexture<uint8_t>(nReactiveMonomers,mStream);
  //initialize the texture with zeroes
  for(auto i=0; i < nReactiveMonomers;i++)
    AAMonomerFlag->host[i] = (uint8_t)0;
    
  miToiNew->pop();
  //if two reactive monomers are connected, they must be sorted into different groups, therefore the move flag...
  uint32_t counter(0);
  for (auto i = 0; i < nReactiveMonomers; i++)
  {
    auto iOld(mNewToOldReactiveID[i]);
    if ( mNeighbors->host[ iOld ].size > 1 )
    {
      counter++;
      for ( size_t j = 0; j < mNeighbors->host[iOld ].size; j++ )
      {
	// the reactive id differ always by more then 1 id and thus, if the difference is abs()=1, then the neighbor is 
	// not a reactive monomer.
	int Diff( mNeighbors->host[ iOld ].neighborIds[j] - iOld );
	if ( !(Diff == 1 || Diff == -1 ) )
	{
	  if (mNeighbors->host[ iOld ].neighborIds[j] < iOld)
	    AAMonomerFlag->host[  miToiNew->host[mNewToOldReactiveID[i]] - mviSubGroupOffsets[ ChainEndSpecies ] ]= (uint8_t)0;
	  else
	    AAMonomerFlag->host[  miToiNew->host[mNewToOldReactiveID[i]] - mviSubGroupOffsets[ ChainEndSpecies ] ]= (uint8_t)1;
	}
      }
    }
  }
  // check if the labels/flags are correctly set (this is just a weak test!!!)
  // every second reactive monomer is set to 1 
  uint32_t counter2(0);
  for(auto i=0; i < nReactiveMonomers;i++)
    if(AAMonomerFlag->host[i] == (uint8_t) 1 ) counter2++; //maybe i must be a uint8_t as a comparison !?
  if (counter != counter2*2)
  {
	  std::stringstream msg;
	  msg << "[" << __FILENAME__ << " set move flags for reactive monomers ] "
	      << "Number of nReactiveMonomers is set to " << nReactiveMonomers 
	      << " and there must be "<< counter << " move flags set to 1 " 
	      << " but only " << counter2 << " have been recorded in the check"  
	      << "!\n";
	  throw std::runtime_error( msg.str() );
  }
  AAMonomerFlag->pushAsync();
  
}

template< typename T_UCoordinateCuda >
void UpdaterGPUScBFM_AA_Connection<T_UCoordinateCuda>::setNrOfReactiveMonomers( T_Id nReactiveMonomers_ )
{
    if ( nReactiveMonomers != 0 )
    {
        std::stringstream msg;
        msg << "[" << __FILENAME__ << "::setNrOfReactiveMonomers] "
            << "Number of nReactiveMonomers already set to           " << nReactiveMonomers << "!\n";
        throw std::runtime_error( msg.str() );
    }
    nReactiveMonomers           = nReactiveMonomers_;
    mLog( "Info" )
	  << "Nr of reactive monomers   "<< nReactiveMonomers <<"\n";
    
};

template< typename T_UCoordinateCuda >
void UpdaterGPUScBFM_AA_Connection<T_UCoordinateCuda>::setReactiveGroup(T_Id monID_, bool reactivity_, T_MaxNumLinks maxNumLinks_){

  
  //fill  mMonomerReactivity
  if (reactivity_ ){
    mNewToOldReactiveID.push_back(monID_);
    D_MonomerReactivity monReact;
    monReact.reactivity=reactivity_;
    monReact.maxNumLinks=maxNumLinks_;
    mMonomerReactivity.push_back(monReact);
  }
  
}

template< typename T_UCoordinateCuda >
void UpdaterGPUScBFM_AA_Connection<T_UCoordinateCuda>::initializeReactiveLattice()
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
void UpdaterGPUScBFM_AA_Connection< T_UCoordinateCuda >::runSimulationOnGPU
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
	      if (!diagMovesOn)  
	      {
		  this-> template launch_CheckSpecies<6>(nBlocks, nThreads, iSpecies, iOffsetLatticeTmp, seed);
		  if ( useCudaMemset )
		      launch_PerformSpeciesAndApply(nBlocks, nThreads, iSpecies, texLatticeTmp );
		  else
		      launch_PerformSpecies(nBlocks,nThreads,iSpecies,texLatticeTmp );
	      }else 
	      {
		  this-> template launch_CheckSpecies<18>(nBlocks, nThreads, iSpecies, iOffsetLatticeTmp, seed);
		  if ( useCudaMemset )
		      launch_PerformSpeciesAndApply(nBlocks, nThreads, iSpecies, texLatticeTmp );
		  else
		      launch_PerformSpecies(nBlocks,nThreads,iSpecies,texLatticeTmp );
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
            }
            else 
            {
	      for(uint32_t n=0; n < 2; n++)
	      {
		if (!diagMovesOn)  
		{
		    this-> template launch_CheckReactiveSpecies<6>(nBlocks, nThreads, iSpecies, iOffsetLatticeTmp, seed, n, AAMonomerFlag->texture );
		    if ( useCudaMemset )
			launch_PerformSpeciesAndApply(nBlocks, nThreads, iSpecies, texLatticeTmp );
		    else
			launch_PerformSpecies(nBlocks,nThreads,iSpecies,texLatticeTmp );
		}else 
		{
		    this-> template launch_CheckReactiveSpecies<18>(nBlocks, nThreads, iSpecies, iOffsetLatticeTmp, seed, n, AAMonomerFlag->texture );
		    if ( useCudaMemset )
			launch_PerformSpeciesAndApply(nBlocks, nThreads, iSpecies, texLatticeTmp );
		    else
			launch_PerformSpecies(nBlocks,nThreads,iSpecies,texLatticeTmp );
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
	
    } // iStep
    
    std::clock_t const t1 = std::clock();
    double const dt = float(t1-t0) / CLOCKS_PER_SEC;
    mLog( "Info" )
    << "run time (GPU): " << nMonteCarloSteps << "\n"
    << "mcs = " << nMonteCarloSteps  << "  speed [performed monomer try and move/s] = MCS*N/t: "
    << nMonteCarloSteps * ( mnAllMonomers / dt )  << "     runtime[s]:" << dt << "\n";
    doCopyBack();
    checkSystem(); // no-op if "Check"-level deactivated
    
}

template< typename T_UCoordinateCuda >
void UpdaterGPUScBFM_AA_Connection< T_UCoordinateCuda >::doCopyBack()
{
    mLog( "Stats" ) << "UpdaterGPUScBFM_AA_Connection< T_UCoordinateCuda >::doCopyBackConnectivity() \n";
    doCopyBackMonomerPositions();
    mLog( "Stats" ) << "UpdaterGPUScBFM_AA_Connection< T_UCoordinateCuda >::doCopyBackConnectivity() \n";
    doCopyBackConnectivity(); // -> need to write a kernel for that. its pretty slow!!! (but works :-) )
}

template< typename T_UCoordinateCuda >
void UpdaterGPUScBFM_AA_Connection< T_UCoordinateCuda >::checkBonds() const
{ 
    /**
     * Check bonds i.e. that |dx|<=3 and whether it is allowed by the given
     * bond set
     */
     std::cout  << "Using UpdaterGPUScBFM_AA_Connection for the checkBonds() \n";  
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
	    if (  ( dx*dx+dy*dy+dz*dz > 10 ) ) erroneousAxis = 3;
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
		    << " at bond number " << iNeighbor  
		    << std::endl;
		throw std::runtime_error( msg.str() );
	    }
	} 
}

template< typename T_UCoordinateCuda >
void UpdaterGPUScBFM_AA_Connection< T_UCoordinateCuda >::checkSystem() const
{
    if ( ! mLog.isActive( "Check" ) )
        return;
        this -> checkLatticeOccupation();
    for (auto i = 0; i < mnAllMonomers; i++)
    {
      if (mGroupIds[i] == 0 )
      {
      	if (mNeighbors->host[i].size ==0 || mNeighbors->host[i].size > 2  )
	{
	  std::stringstream error_message;
	  error_message << "Exceeds the maximum number of bonds of " << 2 << " for crossLinks at monomer Id "
		        <<  i << " with " << mNeighbors->host[i].size << "\n";
	  for (size_t j =0 ; j < mNeighbors->host[i].size; j++ )
	    error_message <<"Neighbor[" <<j << "]= " <<  mNeighbors->host[i].neighborIds[j] << "\n";
	  throw std::runtime_error(error_message.str());
	}
      }
    }
    
    checkBonds();
}

template class UpdaterGPUScBFM_AA_Connection< uint8_t  >;
template class UpdaterGPUScBFM_AA_Connection< uint16_t >;
template class UpdaterGPUScBFM_AA_Connection< uint32_t >;
template class UpdaterGPUScBFM_AA_Connection<  int16_t >;
template class UpdaterGPUScBFM_AA_Connection<  int32_t >;