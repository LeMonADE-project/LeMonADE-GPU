
#include <LeMonADEGPU/utility/GPUConnectionTracker.h>

using ID_t = Tracker::ID_t;

__global__ void kernelTrackBreaks
(
  ID_t           * const dID1      ,
  ID_t           * const dID2      ,
  ID_t             const dSize     ,
  int32_t          const dOffsetA  ,
  int32_t          const dOffsetB  ,
  ID_t           * const diNewToi  ,
  ID_t           * const dOutputID1,
  ID_t           * const dOutputID2 
)
{
  for ( auto i = blockIdx.x * blockDim.x + threadIdx.x;
          i < dSize; i += gridDim.x * blockDim.x )
  {
    auto iMonomer(dID1[i]);
    auto iPartner(dID2[i]);
    if (iPartner == 0 || iMonomer == 0 ) 
    {
      continue; //no Partner found -> go to next Crosslink in the grid 
    }
    iMonomer--;
    iPartner--;
    dID1[i]=0;
    dID2[i]=0;
    dOutputID1[i] = diNewToi[iMonomer + dOffsetA ];
    dOutputID2[i] = ( (diNewToi[iPartner+dOffsetB]+1)<<1 )+0;
//     output[miNewToi[iMonomer-1]-dOffsetA]= ( (miNewToi[iPartner-1]-dOffsetB+1)<<1 )+0;
//     printf("Breaks index=%d %d Id1=%d Id2=%d\n ",diNewToi[iMonomer+dOffsetA], diNewToi[iPartner+dOffsetB], iMonomer+dOffsetA,iPartner+dOffsetB );
  }
}

void Tracker::trackBreaks( ID_t * const ID1     ,
			   ID_t * const ID2     ,
			   ID_t   const size    ,     
			   ID_t * const diNewToi,
			   int32_t const offsetA,
			   int32_t const offsetB,
			   uint32_t const mAge  )
{
  auto nThreads(256);
  auto nBlocks(ceilDiv(size,nThreads));
  kernelTrackBreaks<<<nBlocks,nThreads,0, mStream>>>(
  ID1, 
  ID2, 
  size, 
  offsetA,
  offsetB,
  diNewToi, 
  BondHistoryID1->gpu + counter*nIDs,
  BondHistoryID2->gpu + counter*nIDs
  );  
  age.push_back(mAge);
  increaseCounter();	
  if(counter == bufferSize )
  dumpReactions();
}
__global__ void kernelTrackConnections
(
  ID_t           * const dID1      ,
  ID_t           * const dID2      ,
  ID_t             const dSize     ,
  int32_t          const dOffsetA  ,
  int32_t          const dOffsetB  ,
  ID_t           * const diNewToi  ,
  ID_t           * const dOutputID1,
  ID_t           * const dOutputID2
)
{
  for ( auto i = blockIdx.x * blockDim.x + threadIdx.x;
          i < dSize; i += gridDim.x * blockDim.x )
    {
      auto iMonomer(dID1[i]);
      auto iPartner(dID2[i]);
      dID1[i]=0;
      dID2[i]=0;
      if (iPartner == 0 || iMonomer == 0 ) 
      {
	continue; //no Partner found -> go to next Crosslink in the grid 
      }
      iMonomer--;
      iPartner--;
      dOutputID1[i] = diNewToi[iMonomer + dOffsetA ];
      dOutputID2[i] = ( (diNewToi[iPartner+dOffsetB]+1)<<1 )+1;
//       output[miNewToi[iMonomer-1+offsetA]]= ( (miNewToi[iPartner-1+offsetB]+1)<<1 )+1;
//       printf("Bonds Mon1 = %d  Mon2 = %d Id1=%d Id2=%d %d %d  \n ", iMonomer+dOffsetA, iPartner+dOffsetB, diNewToi[iMonomer+dOffsetA],diNewToi[iPartner+dOffsetB], dOffsetA,dOffsetB );
    }
}
void Tracker::trackConnections( ID_t * const ID1     ,
                                ID_t * const ID2     ,
                                ID_t   const size    ,     
				ID_t * const diNewToi,
				int32_t const offsetA,
				int32_t const offsetB,
				uint32_t const mAge)
{
  auto nThreads(256);	
  auto nBlocks(ceilDiv(size,nThreads));
//   std::cout << "Tracker::trackConnections: offsetA= "<< offset << " mAge= " << mAge  << " size= " << size <<std::endl;
  kernelTrackConnections<<<nBlocks,nThreads,0, mStream>>>(
  ID1, 
  ID2, 
  size, 
  offsetA,
  offsetB,
  diNewToi, 
  BondHistoryID1->gpu + counter*nIDs,
  BondHistoryID2->gpu + counter*nIDs
  );
  age.push_back(mAge);
  increaseCounter();
  if(counter == bufferSize )
  dumpReactions();
}

void Tracker::init(uint32_t bufferSize_, uint32_t nIDs_, cudaStream_t mStream_)
{
  bufferSize=bufferSize_; nIDs=nIDs_; mStream=mStream_;
  BaseClass::setInformationSize(4);
  BaseClass::addComment("MCS Bond/Break ID1 ID2");
  std::cout << "Tracker::init: each BondHistory can take " 
            << 2*bufferSize*nIDs << " number of elements with " 
            << 2*bufferSize*nIDs *sizeof(ID_t)/1024.<< " kB \n";
  BondHistoryID1 = new MirroredVector< ID_t >( 2*bufferSize*nIDs, mStream ); //essentially the ids of the first monomer 
  BondHistoryID2 = new MirroredVector< ID_t >( 2*bufferSize*nIDs, mStream ); //essentially the ids of the second monomer
}
void Tracker::increaseCounter()
{
  counter++;
}


void Tracker::dumpReactions()
{
  BaseClass::setBufferSize(bufferSize);
  BondHistoryID1->popAsync();
  BondHistoryID2->popAsync();
  CUDA_ERROR( cudaStreamSynchronize( mStream ) );
  std::cout <<"counter= " << counter << " nIDs="<<nIDs << std::endl;
  for (uint32_t j=0 ; j < counter ; j ++ ) {
  uint32_t currentAge(age[j]);
    for(uint32_t i =0 ; i < nIDs; i ++)
    {
      auto index(i+nIDs*j);
      auto Mon1(BondHistoryID1->host[index]);
      auto Mon2(BondHistoryID2->host[index]);
      
      if( Mon2  > 0 )
      {
	std::vector<uint32_t> vec;
	vec.push_back(currentAge); //time 
	vec.push_back( Mon2 & 1 ); // either 0 or 1 for remove or add 
	Mon2 = (Mon2 >> 1) -1;
	vec.push_back(std::min(Mon1,Mon2)); 
	vec.push_back(std::max(Mon1,Mon2));
	BaseClass::addConnection(vec);
	BondHistoryID1->host[index]=0;
	BondHistoryID2->host[index]=0;
      }
    }
  }
  BaseClass::dumpReactions();
  counter=0;
  age.resize(0);
  BondHistoryID1->push();
  BondHistoryID2->push();
  
}