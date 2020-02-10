

/*
 * UpdaterGPUScBFM_Tendomers.cu
 *
 *  Created on: 02.12.2019
 *      Authors: Toni Mueller
 */

#include <LeMonADEGPU/updater/UpdaterGPUScBFM_Tendomers.h>
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

using T_Flags            = UpdaterGPUScBFM_Tendomers< uint8_t >::T_Flags;
using T_Lattice          = UpdaterGPUScBFM< uint8_t >::T_Lattice        ;
using T_Label            = UpdaterGPUScBFM_Tendomers< uint8_t >::T_Label;
using T_Id               = UpdaterGPUScBFM_Tendomers< uint8_t >::T_Id;
using T_RingCoordinates  = UpdaterGPUScBFM_Tendomers< uint8_t >::T_RingCoordinates;

__device__ __constant__ uint32_t pitch_d[5];
__device__ __constant__ uint32_t matrixOffset_d[5];
__device__ __constant__ uint32_t subgroupOffset_d[5];
__device__ __constant__ uint32_t dMonomersPerChainP2;  

/**
 * @brief calculates the minimal distances of images for one component 
 * @return int 
 * @param x1 absolute coordinate
 * @param x2 absolute coordinate
 * @param LatticeSize size of the box in the direction of the given coordinates
 */
__device__ __host__  int inline MinImageDistanceComponentForPowerOfTwo(const int x, const uint32_t latticeSize )
{
  //this is only valid for absolute coordinates
  uint32_t latticeSizeM1(latticeSize-1);
  return ( ((x&latticeSizeM1) < (latticeSize/2)) ? (x & latticeSizeM1) :  -(-x & latticeSizeM1));
} 

//kernel for the movement of labels sitting on the monomers of the chains 
/**
 * Maybe the kernel could be speed up by using shared memory. For that the labels of one chain must always 
 * be within one warp. Then the memory loaded for the lattice and the positions could be done by one access!
 * Another performance gain could be to use two arrays: a 0/1 lattice and an ID lattice ?!
 */
template< typename T_UCoordinateCuda >
__global__ void kernelSimulateLabelMoves
(
typename CudaVec4< T_UCoordinateCuda >::value_type
	    const * const __restrict__ dpPolymerSystem        ,
uint32_t            const	       nMonomers              ,
uint64_t            const 	       rSeed                  ,
uint64_t            const 	       rGlobalIteration       ,
BondVectorSet       const 	       checkBondVector        ,
T_RingCoordinates * const 	       dLabelBonds            , 
T_RingCoordinates * const 	       dLabelPosition         ,
T_Id              * const 	       dLatticeLabel          ,
T_Id              * const 	       dpNeighborsMonomer     , 
uint8_t           * const 	       dpNeighborsSizesMonomer,
uint32_t            const              labelOffset
)
{
    uint32_t rn;
    int iGrid = 0;
    for ( auto iMonomer = blockIdx.x * blockDim.x + threadIdx.x;
          iMonomer < nMonomers; iMonomer += gridDim.x * blockDim.x, ++iGrid )
    {
	if ( iGrid % 1 == 0 ) {
	  Saru rng(rGlobalIteration,iMonomer,rSeed);
	  rn =rng.rng32();
        }
        int32_t direction=1-2 *( rn % 2 ); //left(-1) or right (+1)
	// this is a position according to the tendomer id and the curvilinear position along the chain
	// and has nothing to do with the spatial position in 3D
	uint32_t globalLabelID(labelOffset+iMonomer);
        T_RingCoordinates labelPos=dLabelPosition[globalLabelID]; 
        
        uint32_t curvilinearDist=labelPos.x;
        uint32_t chainID=labelPos.y;

        uint32_t latticeIDNew(curvilinearDist+direction + dMonomersPerChainP2*chainID );
	uint32_t latticeID(curvilinearDist  + dMonomersPerChainP2*chainID );
        uint32_t neighborLatticeEntry(dLatticeLabel[latticeIDNew ]);
        
        //neigboring position if not occupied by another label 
        if ( ( neighborLatticeEntry & 1u ) ==  0 ) 
        {
	  //check if ring is connected to another oneBonds
	  uint32_t connectedMonomerID(dLabelBonds[globalLabelID].x);//returns the global monomer ID plus one 
	  if ( connectedMonomerID != 0)
	  {
	    connectedMonomerID--; 
	    uint32_t neigboringMonomer(neighborLatticeEntry>>4);
	    // position of the monomer which gets occupied by the moved label 
	    auto const r0 = dpPolymerSystem[ neigboringMonomer  ]; 
	    // position of the monomer which is connected to the unmoved label 
	    auto const r1 = dpPolymerSystem[ connectedMonomerID  ]; 
	    //if the new bond vector is not in the set, then continue with the next in the grid striding loop 
	    auto const dx(MinImageDistanceComponentForPowerOfTwo(r0.x-r1.x,dcBoxX));
	    auto const dy(MinImageDistanceComponentForPowerOfTwo(r0.y-r1.y,dcBoxY));
	    auto const dz(MinImageDistanceComponentForPowerOfTwo(r0.z-r1.z,dcBoxZ));
	    if ( dx*dx > 9 ) continue; 
	    if ( dy*dy > 9 ) continue; 
	    if ( dz*dz > 9 ) continue; 
	    //only takes dx,dy,dz in the range of [-4:3]
	    if ( checkBondVector(dx,dy,dz) ) continue ;  
	    // still here: establish the new bond and erase the old oneBonds
	    uint32_t connectedLabelID(dLabelBonds[globalLabelID].y>>2);
	    uint32_t connectedLatticeEntry(dLabelPosition[connectedLabelID].x + dMonomersPerChainP2*dLabelPosition[connectedLabelID].y );
	    //ugly!! -_:
	    uint32_t iSpecies                ( (dLatticeLabel[ latticeID ]            & 14u ) >> 1 );
	    uint32_t iSpeciesNeighbor        ( (neighborLatticeEntry                  & 14u ) >> 1 );
	    uint32_t iSpeciesNeighboringLabel( (dLatticeLabel[connectedLatticeEntry ] & 14u ) >> 1 );
	    
	    uint32_t monomerID( dLatticeLabel[ latticeID ] >> 4 );
	    //add bond for the neighboring chain monomer 
	    dpNeighborsMonomer[ matrixOffset_d[iSpeciesNeighbor] + dpNeighborsSizesMonomer[ neigboringMonomer ] * pitch_d[iSpeciesNeighbor] + (neigboringMonomer-subgroupOffset_d[iSpeciesNeighbor]) ] 
	    = connectedMonomerID; 
	    dpNeighborsSizesMonomer[ neigboringMonomer ]++;
	    //change neighbor for the connected label
	    //first search for the bond id and then update the entries
	    dpNeighborsMonomer[ matrixOffset_d[iSpeciesNeighboringLabel] + (dLabelBonds[connectedLabelID].y & 3u) * pitch_d[iSpeciesNeighboringLabel] + (connectedMonomerID- subgroupOffset_d[iSpeciesNeighboringLabel] ) ] 
	    = neigboringMonomer; 

	    //erase bond 
	    //first search for the bond id and then update the entries
	    dpNeighborsSizesMonomer[ monomerID ]--;
	    dpNeighborsMonomer[ matrixOffset_d[iSpecies] + ( dLabelBonds[globalLabelID].y & 3u) * pitch_d[iSpecies] + (monomerID- subgroupOffset_d[iSpecies] ) ] 
	    = dpNeighborsMonomer[ matrixOffset_d[iSpecies] + ( dpNeighborsSizesMonomer[ monomerID ] ) * pitch_d[iSpecies] + (monomerID- subgroupOffset_d[iSpecies] ) ] ;
	    // update the dLabelBonds
	    dLabelBonds[connectedLabelID].x=neigboringMonomer+1;
// 	    dLabelBonds[connectedLabelID].y= stays the same label id and the same bond id ;
// 	    dLabelBonds[globalLabelID].x= is still connected to the same chain id 
	    dLabelBonds[globalLabelID].y= (connectedLabelID  << 2 ) + dpNeighborsSizesMonomer[ neigboringMonomer ]-1 ;
	  }
	  //update lattice 
	  dLatticeLabel[latticeID   ]-=1; //turn non-occupied
	  dLatticeLabel[latticeIDNew]+=1; //turn occupied
	  //update label mLabelPosition
	  dLabelPosition[globalLabelID].x=curvilinearDist+direction;
        }
    }
}


//launch the label move in the kernel 
template< typename T_UCoordinateCuda > 
void UpdaterGPUScBFM_Tendomers<T_UCoordinateCuda>::launch_MoveLabel( 
 const size_t nBlocks, const size_t nThreads, const size_t iSpecies, const uint64_t seed)
{

//   std::cout << "Start kernel call with " << nBlocks << " #blocks and " << nThreads << " #threads. \n";
//   std::cout << "Number of labels in the species group is "  <<   nLabelsPerSpecies[iSpecies] << " at species " << iSpecies << "\n";
  
  kernelSimulateLabelMoves< T_UCoordinateCuda > 
  <<< nBlocks, nThreads, 0, mStream >>>(                
  mPolymerSystemSorted->gpu,
  nLabelsPerSpecies[iSpecies],
  seed,
  hGlobalIterator,
  checkBondVector,
  mLabelBonds->gpu ,
  mLabelPosition->gpu ,
  mLatticeLabel->gpu,
  mNeighborsSorted->gpu,
  mNeighborsSortedSizes->gpu,
  labelOffset[iSpecies]
  );
  hGlobalIterator++;
  CUDA_ERROR( cudaStreamSynchronize( mStream ) );
}


template< typename T_UCoordinateCuda > 
UpdaterGPUScBFM_Tendomers<T_UCoordinateCuda>::UpdaterGPUScBFM_Tendomers():
BaseClass             (      ),
mLatticeLabel         ( NULL ),
mLabelPosition      ( NULL ),
mLabelBonds           ( NULL ),
nMonomersPerChain     ( 0    ),
nTendomers            ( 0    ),
nCrossLinks           ( 0    ),
nLabelsPerTendomerArm ( 0    ),
functionality         ( 0    )
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
void UpdaterGPUScBFM_Tendomers<T_UCoordinateCuda>::destruct(){
      
    DeleteMirroredObject deletePointer;
    deletePointer( mLatticeLabel    , "mLatticeLabel"    );
    deletePointer( mLabelPosition , "mLabelPosition" );
    deletePointer( mLabelBonds      , "mLabelBonds"      );
    if ( deletePointer.nBytesFreed > 0 )
    {
        mLog( "Info" )
            << "Freed a total of "
            << prettyPrintBytes( deletePointer.nBytesFreed )
            << " on GPU and host RAM.\n";
    }
}
template< typename T_UCoordinateCuda > 
UpdaterGPUScBFM_Tendomers<T_UCoordinateCuda>::~UpdaterGPUScBFM_Tendomers()
{
  this->destruct();    
  destruct();
}

template< typename T_UCoordinateCuda >
void UpdaterGPUScBFM_Tendomers<T_UCoordinateCuda>::cleanup()
{
    this->destruct();    
    destruct();
    cudaDeviceSynchronize();
    cudaProfilerStop();
    
}

template < typename T_UCoordinateCuda >
void UpdaterGPUScBFM_Tendomers<T_UCoordinateCuda>::initialize()
{
  
  mLog( "Info" )<<"Cross link functionality is            " << functionality         << "\n";
  mLog( "Info" )<<"Number of cross links is               " << nCrossLinks           << "\n";
  mLog( "Info" )<<"Number of tendomers is                 " << nTendomers            << "\n";
  mLog( "Info" )<<"Number of monomers per tendomer arm is " << nMonomersPerChain     << "\n";
  mLog( "Info" )<<"Number of labels per tendomer arm is   " << nLabelsPerTendomerArm << "\n";
  BaseClass::setAutoColoring(false);
  mLog( "Info" )<< "Start manual coloring of the graph...\n" ;
  //do manual coloring
  /*I assume a tendomer as a building block with m rings threaded between the slip link and one 
  *chain end. (ID with asterik are occupied by a slide ring/slip link)
  *             tendomer(IDs)                      tendomer(colors)
  *   00*-01*-02-03*-04-05-06-07-08-x          A*-B*-A-B*-A-B-A-B-E-...
  *     	   /  			 -->             /
  * x-09-10-11-12*-13*-14-15*-16           ...E-C-D-C-D*-C*-D-C*-D
  */
   
  mGroupIds.resize(mnAllMonomers,4);
  uint32_t ID(0);
  for (uint32_t n=0; n<nTendomers;n++)
  {
    for (uint32_t s=0; s < nMonomersPerChain; s++)
    {
      if (s%2==0)mGroupIds[ID]=0;
      else mGroupIds[ID]=1;
      ID++;
    }
    for (uint32_t s=0; s < nMonomersPerChain; s++)
    {
      if (s%2==0)mGroupIds[ID]=2;
      else mGroupIds[ID]=3;
      ID++;
    }
  }
  // this ensures that no chain label can jump from the chain to the cross link 
  // therefore no attempt should be spend to move the labels on the cross links
//   for ( ID; ID < mnAllMonomers; ID ++ )
// 	mMonomerLabel->host[ID] = std::numeric_limits<T_Label>::max(); 
  mLog( "Info" )<< "odd chains:  \n" ;
  for (uint32_t i =0 ; i < 8; i ++ ) mLog( "Info" )<< "mGroups[" << i << "]= "<< mGroupIds[i] << "\n" ;
  mLog( "Info" )<< "even chains: \n" ;
  for (uint32_t i =nMonomersPerChain ; i < nMonomersPerChain+8; i ++ ) mLog( "Info" )<< "mGroups[" << i << "]= "<< mGroupIds[i] << "\n" ;
  mLog( "Info" )<< "cross links: \n" ;
  for (uint32_t i =nMonomersPerChain*2*nTendomers ; i < nMonomersPerChain*2*nTendomers+8; i ++ ) mLog( "Info" )<< "mGroups[" << i << "]= "<< mGroupIds[i] << "\n" ;
  
  mLog( "Info" )<< "Start manual coloring of the graph...done\n" ;
  mLog( "Info" )<< "Initialize baseclass \n" ;
  BaseClass::initialize();
  
  { decltype( dcBoxX      ) x = mBoxX     ; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxX     , &x, sizeof(x) ) ); }
  { decltype( dcBoxY      ) x = mBoxY     ; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxY     , &x, sizeof(x) ) ); }
  { decltype( dcBoxZ      ) x = mBoxZ     ; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxZ     , &x, sizeof(x) ) ); }
  { decltype( dcBoxXM1    ) x = mBoxXM1   ; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxXM1   , &x, sizeof(x) ) ); }
  { decltype( dcBoxYM1    ) x = mBoxYM1   ; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxYM1   , &x, sizeof(x) ) ); }
  { decltype( dcBoxZM1    ) x = mBoxZM1   ; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxZM1   , &x, sizeof(x) ) ); }
  { decltype( dcBoxXLog2  ) x = mBoxXLog2 ; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxXLog2 , &x, sizeof(x) ) ); }
  { decltype( dcBoxXYLog2 ) x = mBoxXYLog2; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxXYLog2, &x, sizeof(x) ) ); }
  { decltype( nMonomersPerChain      ) x = nMonomersPerChain+2     ; CUDA_ERROR( cudaMemcpyToSymbol( dMonomersPerChainP2     , &x, sizeof(x) ) ); }
  uint32_t tmp_DXTable2[6] = { 0u-2u,2,  0,0,  0,0 };
  uint32_t tmp_DYTable2[6] = {  0,0, 0u-2u,2,  0,0 };
  uint32_t tmp_DZTable2[6] = {  0,0,  0,0, 0u-2u,2 };
  CUDA_ERROR( cudaMemcpyToSymbol( DXTable2_d, tmp_DXTable2, sizeof( tmp_DXTable2 ) ) ); 
  CUDA_ERROR( cudaMemcpyToSymbol( DYTable2_d, tmp_DYTable2, sizeof( tmp_DXTable2 ) ) );
  CUDA_ERROR( cudaMemcpyToSymbol( DZTable2_d, tmp_DZTable2, sizeof( tmp_DXTable2 ) ) );
  mLog( "Info" )<< "Initialize baseclass.done. \n" ;

  nLabels= nTendomers*nLabelsPerTendomerArm*2;
  mLog( "Info" )<< "Total number of labels: "<< nLabels<< "\n" ;
  auto sizeOfLattice = nTendomers*(nMonomersPerChain+2)*2;
  mLog( "Info" )<< "Allocate memory for the label lookup for "<<sizeOfLattice << " lattice entries \n"
		<< "which corresponds to " <<  sizeOfLattice/ sizeof( T_Id)/1024 << " kB.\n";
  mLatticeLabel    = new  MirroredVector< T_Id              > ( sizeOfLattice  );
//   mLabelBonds      = new  MirroredVector< T_RingCoordinates > ( nLabels  );
  mLabelPosition = new  MirroredVector< T_RingCoordinates > ( nLabels  );
  mLog( "Info" )<< "Copy matrix offset, pitch and the subgroup offset for the neighbors and positions into constant memory... \n";
  auto const nSpecies = mnElementsInGroup.size();
  uint32_t tmp_pitch[5]          ;
  uint32_t tmp_matrixOffset[5]   ;
  uint32_t tmp_subgroupOffset[5] ;
  for ( auto j =0 ; j < nSpecies;  j++)
  {
    tmp_subgroupOffset[j] = mviSubGroupOffsets[j];
    tmp_matrixOffset[j]   = mNeighborsSortedInfo.getMatrixOffsetElements(j);
    tmp_pitch[j]          = mNeighborsSortedInfo.getMatrixPitchElements(j);
  }
  CUDA_ERROR( cudaMemcpyToSymbol( pitch_d         , tmp_pitch         , sizeof( tmp_pitch          ) ) ); 
  CUDA_ERROR( cudaMemcpyToSymbol( matrixOffset_d  , tmp_matrixOffset  , sizeof( tmp_matrixOffset   ) ) );
  CUDA_ERROR( cudaMemcpyToSymbol( subgroupOffset_d, tmp_subgroupOffset, sizeof( tmp_subgroupOffset ) ) );
   mLog( "Info" )<< "Copy matrix offset, pitch and the subgroup offset for the neighbors and positions into constant memory:done. \n";
  miToiNew->popAsync(); //needed for the correct init of the lattice 
  CUDA_ERROR( cudaStreamSynchronize( mStream ) );
  vLabelValue.resize(nLabels,0);
  uint32_t ID_R(0) ;
  labelOffset.push_back(0);
  labelOffset.push_back(nTendomers*nLabelsPerTendomerArm);
  nLabelsPerSpecies.push_back(nTendomers*nLabelsPerTendomerArm);
  nLabelsPerSpecies.push_back(nTendomers*nLabelsPerTendomerArm);
  mLabelBonds      = new  MirroredVector< T_RingCoordinates > (  nLabelsPerSpecies[0]+nLabelsPerSpecies[1] );
  std::vector<uint32_t> ciToli(mnAllMonomers,std::numeric_limits<uint32_t>::max());
  
  mLog( "Info" )<< "Start filling lattice, bond table for labels and the 'coordinates' of the labels\n";
  for (uint32_t i =0 ; i < sizeOfLattice; i++ )
  {
    uint32_t curvilinearDist(i % (nMonomersPerChain+2)); // along the chain +2 for one position before and after the chain which marks the ends
    T_Id latticEntry(1);
    if (curvilinearDist != 0 && curvilinearDist != nMonomersPerChain+1)// means at the border of the lattice and thus mark lattic entry as occupied
    {
      uint32_t ChainID(( i-curvilinearDist ) /( nMonomersPerChain+2 ));  //even ID + next odd ID belong to the same tendomer
      uint32_t OldID(curvilinearDist-1+ChainID*nMonomersPerChain);
      // lattice contain 
      // 28 bit  | 3 bit    | 1 bit 
      // chainID | iSpecies | Occupied 
      // chainID  = latticeEntry >>4 
      // (iSpecies & 14u)>>1
      // Occupied & 1u 
      latticEntry = (miToiNew->host[OldID]<<4 )+ (mGroupIds[OldID]<<1);
      if (i <20 )
	std::cout <<"i="<<i<<" OldID " << OldID<<" c=" << mGroupIds[OldID] << " " << miToiNew->host[OldID]<< "\n";
      if (vMonomerLabel[OldID] > 0 
       && vMonomerLabel[OldID] != std::numeric_limits<uint32_t>::max() 
       )
      {
	//I assume to have a better cache hit rate with the following sorting of the ideas 
	uint32_t sortedID; // 0 2 4 6 8 10 .... offset 1 3 5 7 9 
// 	sortedID=ID_R;
	if (nLabelsPerTendomerArm %2 == 1 ){
	  if ( ID_R % 2 == 0 )
	    sortedID=ID_R/2;
	  else
	    sortedID=nLabelsPerSpecies[1]+((ID_R-1)/2);
	}else 
	{
	  if ( ( ID_R % 2 ) == 0 && ( ChainID % 2 ) == 0 ) //A-Type 
	    sortedID=ID_R/2;
	  else if ( ( ID_R % 2 ) == 1 && ( ChainID % 2 ) == 1 ) //A-Type 
	    sortedID=(ID_R-1)/2;
	  else if ( ( ID_R % 2 ) == 1 && ( ChainID % 2 ) == 0 ) //B-Type 
	    sortedID=nLabelsPerSpecies[1]+((ID_R-1)/2);
	  else	//if ( ( ID_R % 2 ) == 0 && ( ChainID % 2 ) == 1 ) B-Type 
	    sortedID=nLabelsPerSpecies[1]+(ID_R/2);
	}
	ciToli[miToiNew->host[OldID]]=sortedID;
	//curvilinear distance along the chain
	mLabelPosition->host[sortedID].x = curvilinearDist;
	//chain ID
	mLabelPosition->host[sortedID].y = ChainID;   
	//needed to copy back the labels (with their value) correctly 
	vLabelValue[sortedID]=vMonomerLabel[OldID];
	//changes the last bit to 1 und thus marks it as occupied
	latticEntry++;
	uint32_t nNeighbors = (uint32_t) mNeighbors->host[OldID].size; 
	uint32_t iGlobaRingNeighbor(std::numeric_limits<uint32_t>::max());
	uint32_t BondID(0);
	for ( size_t j = 0; j < nNeighbors; j++ )
	{
	  uint32_t NeighborID ( mNeighbors->host[OldID].neighborIds[j] ) ;
	  if (   vMonomerLabel[ NeighborID] != vMonomerLabel[OldID] // label is not of the same chain 
	      && vMonomerLabel[ NeighborID] != 0                // is just an empty label of the chain
	      && vMonomerLabel[ NeighborID] != std::numeric_limits<uint32_t>::max() //is not the cross linker 
	    )
	  {
	      iGlobaRingNeighbor = NeighborID; 
	      BondID = j; 
	      break;
	  }
	}
 	if ( iGlobaRingNeighbor  != std::numeric_limits<uint32_t>::max())
	{
	  mLabelBonds->host[sortedID].x=miToiNew->host[iGlobaRingNeighbor]+1; //slip link  
	  //at this point not all label ids are distributed and thus 
	  //the current label is maybe connected to one which a was not
	  //visited upt to now
	  //therefore just store the bondID in the hope that it gets not change
	  //when the monomers get resorted... -_-
	  mLabelBonds->host[sortedID].y=BondID;
	}
	else{ 
	  mLabelBonds->host[sortedID].x=0; //slide ring
	  mLabelBonds->host[sortedID].y=0;
	}
	ID_R++;
      }
    }
    //fill lattice with the information of occupation and the new global monomer ID of the chain 
    mLatticeLabel->host[i] = latticEntry;  
  }
//   for(size_t i=0; i <nMonomersPerChain+2; i++)
//     std::cout << ( (mLatticeLabel->host[i]) & 1u )<<"\t";
//   std::cout <<"\n";
//   for(size_t i=0; i <nMonomersPerChain+2; i++)
//     std::cout << ( (mLatticeLabel->host[i]) & 14u )<<"\t";
//   std::cout <<"\n";  
//     for(size_t i=0; i <nMonomersPerChain+2; i++)
//     std::cout << ( (mLatticeLabel->host[i]) >>4 )<<"\t";
//   std::cout <<"\n";  
  if ( ID_R != nLabels )
  {
    std::stringstream error_message;
    error_message << "The visited labels " << ID_R<< " does not agree with the calculated number " << nLabels<< "\n";
    throw std::runtime_error(error_message.str());
  }
  for(size_t i=0; i <20; i++)
    std::cout << "i=" << i << " " << mLabelPosition->host[i].x << " " << mLabelPosition->host[i].y << "\n";
  miNewToi->popAsync();
    CUDA_ERROR( cudaStreamSynchronize( mStream ) );
  for(size_t i=0; i <20; i++)
    std::cout << "i=" << i << " " << ( (mLatticeLabel->host[i]) & 14u ) << " " << (mLatticeLabel->host[i]>>4) << " " << mGroupIds[miNewToi->host[mLatticeLabel->host[i]>>4]]<<"\n" ;

  mLog( "Info" )<< "Start filling lattice, bond table for labels and the 'coordinates' of the labels:done.\n";
  for(auto i=0; i < nLabels; i ++)
  {
    mLabelBonds->host[i].y= (ciToli[mLabelBonds->host[i].x-1] << 2 ) + mLabelBonds->host[i].y; 
  }
  
  mLatticeLabel->pushAsync();
  mLabelBonds->pushAsync();
  mLabelPosition->pushAsync();
//   CUDA_ERROR( cudaStreamSynchronize( mStream ) );
  
}
template< typename T_UCoordinateCuda >
void UpdaterGPUScBFM_Tendomers<T_UCoordinateCuda>::setNTendomers             ( uint32_t nTendomers_            )
{
    if ( nTendomers != 0 )
    {
        std::stringstream msg;
        msg << "[" << __FILENAME__ << "::setNTendomers] "
            << "Number of tendomers already set to " << nTendomers << "!\n";
        throw std::runtime_error( msg.str() );
    }
    nTendomers = nTendomers_;
    mLog( "Info" ) << "Nr of tendomers    "<< nTendomers  <<"\n";
}
template< typename T_UCoordinateCuda >
void UpdaterGPUScBFM_Tendomers<T_UCoordinateCuda>::setNumCrossLinkers        ( uint32_t nCrossLinks_           )
{
    if ( nCrossLinks != 0 )
    {
        std::stringstream msg;
        msg << "[" << __FILENAME__ << "::setNumCrossLinkers] "
            << "Number of crosslinks already set to " << nCrossLinks << "!\n";
        throw std::runtime_error( msg.str() );
    }
    nCrossLinks= nCrossLinks_;
    mLog( "Info" ) << "Nr of crosslinks   "<< nCrossLinks <<"\n";
}
template< typename T_UCoordinateCuda >
void UpdaterGPUScBFM_Tendomers<T_UCoordinateCuda>::setNumMonomersPerChain    ( uint32_t nMonomersPerChain_     )
{
    if ( nMonomersPerChain != 0 )
    {
        std::stringstream msg;
        msg << "[" << __FILENAME__ << "::setNumMonomersPerChain] "
            << "Number of monomers per chain already set to           " << nMonomersPerChain << "!\n";
        throw std::runtime_error( msg.str() );
    }
    nMonomersPerChain = nMonomersPerChain_;
    mLog( "Info" ) << "Nr of monomer per chain "<< nMonomersPerChain <<"\n";
}
template< typename T_UCoordinateCuda >
void UpdaterGPUScBFM_Tendomers<T_UCoordinateCuda>::setNumLabelsPerTendomerArm( uint32_t nLabelsPerTendomerArm_ )
{
    if ( nLabelsPerTendomerArm != 0 )
    {
        std::stringstream msg;
        msg << "[" << __FILENAME__ << "::setNumLabelsPerTendomerArm] "
            << "Number of labels per tendomer arm already set to " << nLabelsPerTendomerArm << "!\n";
        throw std::runtime_error( msg.str() );
    }
    nLabelsPerTendomerArm = nLabelsPerTendomerArm_;
    mLog( "Info" ) << "Nr of labels per tendomer arm "<< nLabelsPerTendomerArm <<"\n";
//     mMonomerLabel = new MirroredVector< T_Label  >( mnAllMonomers );
  // this ensures that no chain label can jump from the chain to the cross link 
  // therefore no attempt should be spend to move the labels on the cross links
    vMonomerLabel.resize(mnAllMonomers, std::numeric_limits<uint32_t>::max());
}
template< typename T_UCoordinateCuda >
void UpdaterGPUScBFM_Tendomers<T_UCoordinateCuda>::setFunctionality          ( uint32_t functionality_ )
{
    if ( functionality != 0 )
    {
        std::stringstream msg;
        msg << "[" << __FILENAME__ << "::setFunctionality] "
            << "Functionality of crosslinks already set to " << functionality << "!\n";
        throw std::runtime_error( msg.str() );
    }
    functionality = functionality_;
    mLog( "Info" ) << "Functionality of crosslinks "<< functionality <<"\n";
}

template< typename T_UCoordinateCuda >
void UpdaterGPUScBFM_Tendomers<T_UCoordinateCuda>::setLabel( uint32_t ID_, uint32_t label_){vMonomerLabel[ID_]=label_;}
template< typename T_UCoordinateCuda >
int32_t UpdaterGPUScBFM_Tendomers<T_UCoordinateCuda>::getLabel( uint32_t ID_)
{ 
  int32_t  label=vMonomerLabel[ID_];
  if (label == std::numeric_limits<uint32_t>::max())
    label=0;
  return label;
}

template< typename T_UCoordinateCuda  >
void UpdaterGPUScBFM_Tendomers< T_UCoordinateCuda >::runSimulationOnGPU
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
                //move labels 
        for ( uint32_t iSubStep = 0; iSubStep < labelOffset.size(); ++iSubStep ) 
	{
            /* randomly choose which monomer group to advance */
//             auto const iSpecies = randomNumbers.r250_rand32() % (labelOffset.size());
            auto const iSpecies = iSubStep;
            auto const nThreads = 128;
            auto const nBlocks  = ceilDiv( nLabelsPerSpecies[iSpecies], nThreads );
            auto const seed     = randomNumbers.r250_rand32();
	    //move label 
	    launch_MoveLabel(nBlocks, nThreads, iSpecies, seed);
        } // iSubstep
//         mLog( "Check") << "Check after moving the labels\n";
//         doCopyBack();
// 	checkSystem(); // no-op if "Check"-level deactivated
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

	    if ( useCudaMemset )
	    {
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
        } // iSubstep
//         mLog( "Check") << "Check after moving the positions\n";
//         doCopyBack();
// 	checkSystem(); // no-op if "Check"-level deactivated


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
void UpdaterGPUScBFM_Tendomers< T_UCoordinateCuda >::doCopyBackLabels()
{
    mLabelPosition->pop(false);
   
    CUDA_ERROR( cudaStreamSynchronize( mStream ) ); // finish e.g. initializations
    vMonomerLabel.resize(0);
    vMonomerLabel.resize(mnAllMonomers, std::numeric_limits<uint32_t>::max());
    for ( auto i=0; i < nLabels ; i ++)
    {
//       std::cout << "i=" << i << "\t"
// 		<< "ID=" << mLabelPosition->host[i].x-1 + nMonomersPerChain*mLabelPosition->host[i].y << "\t" 
// 		<< "s=" << mLabelPosition->host[i].x << "\t" 
// 		<< "cID=" <<mLabelPosition->host[i].y 
// 		<<std::endl; 
      vMonomerLabel[ mLabelPosition->host[i].x-1 + nMonomersPerChain*mLabelPosition->host[i].y ]=vLabelValue[i];
    }
}

template< typename T_UCoordinateCuda >
void UpdaterGPUScBFM_Tendomers< T_UCoordinateCuda >::doCopyBack()
{
    mLog( "Stats" ) << "UpdaterGPUScBFM_Tendomers< T_UCoordinateCuda >::doCopyBackMonomerPositions() \n";
    doCopyBackMonomerPositions();
    mLog( "Stats" ) << "UpdaterGPUScBFM_Tendomers< T_UCoordinateCuda >::doCopyBackConnectivity() \n";
    doCopyBackConnectivity(); // -> need to write a kernel for that. its pretty slow!!! (but works :-) )
    mLog( "Stats" ) << "UpdaterGPUScBFM_Tendomers< T_UCoordinateCuda >::doCopyBackLabels() \n";
    doCopyBackLabels();
}

template< typename T_UCoordinateCuda >
void UpdaterGPUScBFM_Tendomers< T_UCoordinateCuda >::checkBonds() const
{ 
    /**
     * Check bonds i.e. that |dx|<=3 and whether it is allowed by the given
     * bond set
     */
     std::cout  << "Using UpdaterGPUScBFM_Tendomers for the checkBonds() \n";  
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
void UpdaterGPUScBFM_Tendomers< T_UCoordinateCuda >::checkSystem() const
{
    if ( ! mLog.isActive( "Check" ) )
        return;
    BaseClass::checkLatticeOccupation();
//     for (auto i = 0; i < mnAllMonomers; i++)
//     {
//       if (mGroupIds[i] == 0 ) 
//       {
//       	if (mNeighbors->host[i].size ==0 || mNeighbors->host[i].size > 2  )
// 	{
// 	  std::stringstream error_message;
// 	  error_message << "Exceeds the maximum number of bonds of " << 2 << " for crossLinks at monomer Id "
// 		        <<  i << " with " << mNeighbors->host[i].size << "\n";
// 	  for (size_t j =0 ; j < mNeighbors->host[i].size; j++ )
// 	    error_message <<"Neighbor[" <<j << "]= " <<  mNeighbors->host[i].neighborIds[j] << "\n";
// 	  throw std::runtime_error(error_message.str());
// 	}
//       }
//     }
    
    checkBonds();
}

template class UpdaterGPUScBFM_Tendomers< uint8_t  >;
template class UpdaterGPUScBFM_Tendomers< uint16_t >;
template class UpdaterGPUScBFM_Tendomers< uint32_t >;
template class UpdaterGPUScBFM_Tendomers<  int16_t >;
template class UpdaterGPUScBFM_Tendomers<  int32_t >;