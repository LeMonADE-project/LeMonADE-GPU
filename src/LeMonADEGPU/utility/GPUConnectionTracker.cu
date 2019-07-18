
#include <LeMonADEGPU/utility/GPUConnectionTracker.h>

using ID_t = Tracker::ID_t;

__global__ void kernelTrackBreaks
(
  ID_t     * const dBreaks ,
  uint32_t   const size    ,
  ID_t     * const miNewToi,
  ID_t     * const output  ,
  uint32_t   const offset  
)
{
  for ( auto i = blockIdx.x * blockDim.x + threadIdx.x;
          i < size; i += gridDim.x * blockDim.x )
    {
      auto iPartner(dBreaks[i]);
      if (iPartner == 0 ) 
      {
	dBreaks[i]=0;
	continue; //no Partner found -> go to next Crosslink in the grid 
      }
      output[miNewToi[i-1]-offset]= ( (miNewToi[iPartner-1]+1)<<1 )+0;
//       printf("Breaks index=%d  %d %d Id1=%d Id2=%d\n ",miNewToi[i-1]-offset, miNewToi[i-1], miNewToi[iPartner-1], iPartner-1, i-1  );
      dBreaks[i]=0;
    }
}

void Tracker::trackBreaks( ID_t     * const dBreaks, 
			   uint32_t   const size, 
			   ID_t     * const miNewToi,
			   uint32_t   const mAge){

  auto nThreads(256);
  auto nBlocks(ceilDiv(size,nThreads));
  kernelTrackBreaks<<<nBlocks,nThreads,0, mStream>>>(
  dBreaks, 
  size, 
  miNewToi, 
  BondHistory->gpu + counter*nIDs, 
  IDoffset);  
  age.push_back(mAge);
  increaseCounter();	

}
__global__ void kernelTrackConnections
(
  ID_t    const  * const mCrossLinkFlags,
  ID_t    const  * const mCrossLinkIDS  ,
  ID_t       const size           ,
  uint32_t   const chainoffset    ,
  ID_t     * const miNewToi       ,
  ID_t     * const output         ,
  uint32_t   const offset         
)
{
  for ( auto i = blockIdx.x * blockDim.x + threadIdx.x;
          i < size; i += gridDim.x * blockDim.x )
    {
      auto iPartner(mCrossLinkFlags[i]);
      auto iMonomer(mCrossLinkIDS[i]);
      if (iPartner == 0 || iMonomer == 0 ) 
      {
	continue; //no Partner found -> go to next Crosslink in the grid 
      }
      output[miNewToi[iMonomer-1]-offset]= ( (miNewToi[iPartner-1+chainoffset]+1)<<1 )+1;
//       printf("Bonds Mon1 = %d  Mon2 = %d Id1=%d Id2=%d %d %d  \n ", iPartner-1+chainoffset, iMonomer-1, miNewToi[iMonomer-1],miNewToi[iPartner-1+chainoffset], offset, chainoffset );
    }
}
void Tracker::trackConnections( ID_t * const mCrossLinkFlags,
                                ID_t * const mCrossLinkIDS  ,
                                ID_t   const size  ,     
				ID_t * const miNewToi,
				uint32_t const offset,
				uint32_t const mAge)
{
  auto nThreads(256);	
  auto nBlocks(ceilDiv(size,nThreads));
  
  kernelTrackConnections<<<nBlocks,nThreads,0, mStream>>>(
  mCrossLinkFlags, 
  mCrossLinkIDS, 
  size, 
  offset,
  miNewToi, 
  BondHistory->gpu + counter *nIDs, 
  IDoffset);
  age.push_back(mAge);
  increaseCounter();
}

void Tracker::init(uint32_t bufferSize_, uint32_t nIDs_, cudaStream_t mStream_)
{
  bufferSize=bufferSize_; nIDs=nIDs_; mStream=mStream_;
  BaseClass::setInformationSize(4);
  BaseClass::addComment("MCS Bond/Break ID1 ID2");
  std::cout << "Tracker::init: BondHistory can take " 
            << 2*bufferSize*nIDs << " number of elements with " 
            << 2*bufferSize*nIDs *sizeof(ID_t)/1024.<< " kB \n";
  BondHistory = new MirroredVector< ID_t >( 2*bufferSize*nIDs, mStream );
}
void Tracker::increaseCounter()
{
  counter++;
  if(counter == bufferSize ){
//   std::cout << "Tracker::increaseCounter()  increased counter and"
// 	    << " start dumpReactions, because the bufferSize "
//             << counter << "/" << bufferSize<< "is reached.\n "; 
    dumpReactions();
    }
}


void Tracker::dumpReactions()
{
  BaseClass::setBufferSize(bufferSize);
  BondHistory->popAsync();
  CUDA_ERROR( cudaStreamSynchronize( mStream ) );

  for (uint32_t j=0 ; j < counter ; j ++ ) {
  uint32_t currentAge(age[j]);
    for(uint32_t i =0 ; i < nIDs; i ++)
    {
      auto Mon(i+j*nIDs);
      uint32_t entry(BondHistory->host[Mon]);
      if( entry  > 0 )
      {
	std::vector<uint32_t> vec;
	vec.push_back(currentAge); //time 
	vec.push_back(entry & 1 ); // either 0 or 1 for remove or add 
	uint32_t ID1(i+IDoffset);
	uint32_t ID2((BondHistory->host[Mon] >> 1) -1);
	vec.push_back(std::min(ID1,ID2)); 
	vec.push_back(std::max(ID1,ID2));
	BaseClass::addConnection(vec);
	BondHistory->host[Mon]=0;
      }
    }
  }
  BaseClass::dumpReactions();
  counter=0;
  age.resize(0);
  BondHistory->push();
  
}